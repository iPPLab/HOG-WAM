"""TWP —— L3 训练层的一集一文件格式。**读写的唯一真相**。

建库工具和训练 loader 都 import 这个模块。任何一方自己解析字节 = 两份真相,
而这个项目栽过的坑全是"两份真相悄悄漂了但不报错"。

零依赖(只要 numpy),因为 TPU 上的环境是钉死的:没有 pyarrow、没有 pickle。
序列化一律 ``tobytes()`` / ``frombuffer()`` —— 用 pickle/tfds.features 存变长曾经
让别人的吞吐从 130 样本/秒掉到 3(45×)。

布局::

    preamble 40 B   magic 'TWP3' | version u32 | n_sections u32 | episode_id u32
                    n_frames u32 | n_cams u32  | action_dim u32
                    text_rows u32 | text_len u32 | text_dim u32
    段表     N×20 B  {tag u32, offset u64, length u64}
    各段     按段表顺序紧邻排列,覆盖文件其余部分

段(**按此顺序排列,不是按 tag 大小**)::

    1 LENGTHS   int32[T*C]        每张 JPEG 的字节数,帧主序 f0c0 f0c1 f0c2 f1c0 ...
    2 ACTIONS   f32[T, action_dim]  **原值**。delta 与归一化是 loader 的事,不烘进盘
    3 SEEN      训练指令
    6 TEXT_EMB  fp16[text_rows, text_len, text_dim]  本集指令的 umT5 编码
    4 PAYLOAD   JPEG 字节,帧主序
    5 UNSEEN    评测指令 —— **训练路径禁止读**,见 payload_end()

**v3 相对 v2**:加 TEXT_EMB 段,且**放在 PAYLOAD 之前**。训练侧本来就只读
``[0, payload_end)``,所以 embedding 落在这一次读取里 —— 零额外请求;而 unseen
仍在 payload 之后,照样碰不到。文本几何进 preamble,同样是"文件自己说清自己的形状"。

分层的道理:规范版(按指令文本去重)在 **L2**,一次编好永不再编;L3 只是把本集要用的
那几行**烘进来**,换口径就重建 L3,不用重新编码。所以这不违反"L3 可丢弃"这条。

**v2 相对 v1**:形状(T / C / action_dim)进 preamble。v1 靠 ``actions段长 ÷ (4×14)``
反推 T、再靠 T 反推 C —— 两者都建立在"动作 14 维"这个硬编码上,换一个数据源就崩,
而且是**算出一个错的 T 然后静默出错**,不是报错。

不在文件里的东西(全部是运行时的事):**画布布局**、latent、delta/归一化。
换 VAE / 换分辨率 / 换画布 / 换 F,stride / 换归一化口径 —— 都不用重建。
(换文本编码器要重建 L3,但**不用重新编码**:规范版在 L2。)

相机的**物理角色**(哪一路是头、哪一路是左腕)是数据事实,记在数据集级的 ``spec.json``
的 ``camera_roles`` 里,与 ``cameras`` 一一对应;**画布怎么拼**是代码的事,不在数据里。
"""
from __future__ import annotations

import struct
from typing import Sequence

import numpy as np

MAGIC = b"TWP3"
VERSION = 3
PREAMBLE = 40
ENTRY = 20                      # 段表一条 = tag u32 + offset u64 + length u64

LENGTHS, ACTIONS, SEEN, PAYLOAD, UNSEEN, TEXT_EMB = 1, 2, 3, 4, 5, 6
STATES = 7
SECTION_NAMES = {LENGTHS: "lengths", ACTIONS: "actions", SEEN: "instr_seen",
                 TEXT_EMB: "text_emb", PAYLOAD: "payload", UNSEEN: "instr_unseen", STATES: "states"}


# ---------------------------------------------------------------- 写

def pack_strings(items: Sequence[str]) -> bytes:
    """count u32 | length u32[count] | UTF-8 拼接。定长表 + 裸字节,不是序列化容器。"""
    enc = [s.encode("utf-8") for s in items]
    return (struct.pack("<I", len(enc))
            + b"".join(struct.pack("<I", len(e)) for e in enc)
            + b"".join(enc))


def build(episode_id: int, jpeg_lengths, jpeg_payload: bytes,
          actions: np.ndarray, seen: Sequence[str], unseen: Sequence[str],
          *, n_cams: int, text_emb: np.ndarray | None = None, states: np.ndarray | None = None) -> bytes:
    """组装一集。调用方保证 jpeg_payload 是帧主序、与 jpeg_lengths 对齐。

    text_emb: [text_rows, text_len, text_dim] fp16,本集要用的指令编码。
    None = 不带文本(段仍然存在但长度为 0,让段序在有无文本时保持一致)。
    """
    lengths = np.asarray(jpeg_lengths, dtype=np.int32)
    acts = np.ascontiguousarray(actions, dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError(f"actions 应为 (T, dim),实为 {acts.shape}")
    T, dim = acts.shape
    if len(lengths) != T * n_cams:
        raise ValueError(f"lengths {len(lengths)} != T*C = {T}*{n_cams}")
    if int(lengths.sum()) != len(jpeg_payload):
        raise ValueError(f"lengths 和 {lengths.sum()} != payload {len(jpeg_payload)}")
    if text_emb is None:
        t_rows = t_len = t_dim = 0
        t_blob = b""
    else:
        te = np.ascontiguousarray(text_emb, dtype=np.float16)
        if te.ndim != 3:
            raise ValueError(f"text_emb 应为 (rows, len, dim),实为 {te.shape}")
        t_rows, t_len, t_dim = te.shape
        t_blob = te.tobytes()

    # 段序:TEXT_EMB 必须在 PAYLOAD **之前** —— 训练侧只读 [0, payload_end),
    # 放后面就读不到,放到 UNSEEN 之后更是会把评测指令一起拖进训练进程。
    secs = [(LENGTHS, lengths.tobytes()), (ACTIONS, acts.tobytes()),
            (SEEN, pack_strings(seen)), (TEXT_EMB, t_blob),
            (PAYLOAD, jpeg_payload), (UNSEEN, pack_strings(unseen))]
    if states is not None:
        state = np.ascontiguousarray(states, dtype=np.float32)
        if state.shape != acts.shape:
            raise ValueError("states must match the action array shape")
        secs.insert(2, (STATES, state.tobytes()))
    pre = MAGIC + struct.pack("<IIIIIIIII", VERSION, len(secs), episode_id, T, n_cams, dim,
                              t_rows, t_len, t_dim)
    off = PREAMBLE + len(secs) * ENTRY
    table, body = b"", b""
    for tag, blob in secs:
        table += struct.pack("<IQQ", tag, off, len(blob))
        body += blob
        off += len(blob)
    return pre + table + body


# ---------------------------------------------------------------- 读

class Episode:
    """对一段 .twp 字节的零拷贝视图。切片全是指针运算,不复制不解码。"""

    __slots__ = ("buf", "episode_id", "version", "n_frames", "n_cams", "action_dim",
                 "text_rows", "text_len", "text_dim", "_sec", "_cum", "_seen")

    def __init__(self, buf, *, expect_id: int | None = None):
        if bytes(buf[:4]) != MAGIC:
            raise ValueError(f"magic 不符: {bytes(buf[:4])!r}(需要 {MAGIC!r})")
        (self.version, n_sec, self.episode_id, self.n_frames, self.n_cams,
         self.action_dim, self.text_rows, self.text_len,
         self.text_dim) = struct.unpack("<IIIIIIIII", buf[4:PREAMBLE])
        if self.version != VERSION:
            raise ValueError(f"版本 {self.version} != {VERSION}")
        # 抓"集错位":文件名对但内容是别的集。判定门之外的运行时防线
        if expect_id is not None and self.episode_id != expect_id:
            raise ValueError(f"集号错位:文件内 {self.episode_id},期望 {expect_id}")
        self.buf = buf
        self._sec = {}
        for k in range(n_sec):
            tag, off, ln = struct.unpack("<IQQ", buf[PREAMBLE + k*ENTRY: PREAMBLE + (k+1)*ENTRY])
            self._sec[tag] = (off, ln)
        self._cum = None
        self._seen = None
        # 形状自洽:preamble 说的和段长必须对得上,否则文件是坏的
        if self._sec[ACTIONS][1] != self.n_frames * self.action_dim * 4:
            raise ValueError(f"ep{self.episode_id}: actions 段 {self._sec[ACTIONS][1]}B "
                             f"!= T*dim*4 = {self.n_frames}*{self.action_dim}*4")
        if self._sec[LENGTHS][1] != self.n_frames * self.n_cams * 4:
            raise ValueError(f"ep{self.episode_id}: lengths 段 {self._sec[LENGTHS][1]}B "
                             f"!= T*C*4 = {self.n_frames}*{self.n_cams}*4")
        if STATES in self._sec:
            off, length = self._sec[STATES]
            if length != self._sec[ACTIONS][1] or off + length > self._sec[PAYLOAD][0]:
                raise ValueError("invalid aligned states section")
        want_t = self.text_rows * self.text_len * self.text_dim * 2
        if TEXT_EMB in self._sec and self._sec[TEXT_EMB][1] != want_t:
            raise ValueError(f"ep{self.episode_id}: text_emb 段 {self._sec[TEXT_EMB][1]}B "
                             f"!= rows*len*dim*2 = {self.text_rows}*{self.text_len}"
                             f"*{self.text_dim}*2")
        # TEXT_EMB 必须在 PAYLOAD 之前,否则训练侧的一次读取拿不到它
        if TEXT_EMB in self._sec and self._sec[TEXT_EMB][0] > self._sec[PAYLOAD][0]:
            raise ValueError(f"ep{self.episode_id}: text_emb 段在 payload 之后 —— "
                             f"训练侧只读 [0,payload_end),会读不到")

    # -------- 段访问

    def _raw(self, tag: int) -> memoryview:
        off, ln = self._sec[tag]
        return memoryview(self.buf)[off:off + ln]

    @property
    def lengths(self) -> np.ndarray:
        off, ln = self._sec[LENGTHS]
        return np.frombuffer(self.buf, np.int32, ln // 4, off)

    @property
    def actions(self) -> np.ndarray:
        """**原值**。delta 与归一化不在这里做 —— 调 tinywam 现有那段,别重写。"""
        off, _ = self._sec[ACTIONS]
        return np.frombuffer(self.buf, np.float32,
                             self.n_frames * self.action_dim, off).reshape(self.n_frames, self.action_dim)

    @property
    def states(self) -> np.ndarray | None:
        if STATES not in self._sec:
            return None
        off, _ = self._sec[STATES]
        return np.frombuffer(self.buf, np.float32, self.n_frames * self.action_dim, off).reshape(
            self.n_frames, self.action_dim)

    @property
    def seen(self) -> list[str]:
        # 缓存:一集在池里活很久,每步每被抽到都重解 100 条串是白烧 CPU
        if self._seen is None:
            self._seen = unpack_strings(self._raw(SEEN))
        return self._seen

    @property
    def unseen(self) -> list[str]:
        """**训练路径禁止调用。** 官方规定训练用 seen、评测用 unseen;
        混用会毁掉语言泛化基准,而且毁了之后无法自证清白。"""
        return unpack_strings(self._raw(UNSEEN))

    @property
    def text_emb(self) -> np.ndarray:
        """[text_rows, text_len, text_dim] fp16。零拷贝视图,不复制。

        没有文本段(或长度 0)时返回形状 (0,0,0) 的空数组,让调用方能明确判断,
        而不是拿到一个看起来正常的零向量。
        """
        if TEXT_EMB not in self._sec or self.text_rows == 0:
            return np.empty((0, 0, 0), np.float16)
        off, ln = self._sec[TEXT_EMB]
        return np.frombuffer(self.buf, np.float16, ln // 2, off).reshape(
            self.text_rows, self.text_len, self.text_dim)

    def payload_end(self) -> int:
        """训练侧应当只 range-read [0, payload_end)。unseen 段在此之后。"""
        off, ln = self._sec[PAYLOAD]
        return off + ln

    # -------- 帧访问

    def _offsets(self) -> np.ndarray:
        if self._cum is None:                       # 每集一次,微秒级
            self._cum = np.concatenate([[0], np.cumsum(self.lengths, dtype=np.int64)])
        return self._cum

    def jpeg(self, frame: int, cam: int) -> memoryview:
        """第 frame 帧第 cam 路的 JPEG 原始字节(零拷贝)。"""
        base = self._sec[PAYLOAD][0]
        cum = self._offsets()
        k = frame * self.n_cams + cam
        return memoryview(self.buf)[base + cum[k]: base + cum[k + 1]]


def unpack_strings(b) -> list[str]:
    b = bytes(b)
    n = struct.unpack("<I", b[:4])[0]
    lens = struct.unpack(f"<{n}I", b[4:4 + 4 * n])
    out, p = [], 4 + 4 * n
    for L in lens:
        out.append(b[p:p + L].decode("utf-8"))
        p += L
    return out
