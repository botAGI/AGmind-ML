"""Document segmenter — единый источник правды для data-gen И инференса.

Разбивает markdown на нумерованные атомарные юниты для context-aware сплиттера:
проза → предложения (razdel), а таблицы / блоки кода / заголовки → единые
атомарные юниты. Сплиттер-модель оперирует НОМЕРАМИ этих юнитов, поэтому
целостность таблиц обеспечивает именно сегментатор (модель не режет внутри юнита).

Один и тот же `segment_units` используется при генерации обучающих данных и на
инференсе — это убирает train/serve skew (модель видит юниты той же формы, что
учили). Возвращает список кортежей (type, text), type ∈ {sent, table, code, head}.
"""
import re
from razdel import sentenize

_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _is_sep(s: str) -> bool:
    """Строка-разделитель markdown-таблицы, напр. `|---|:--:|---|`."""
    return bool(_SEP_RE.match(s)) and "-" in s


def _is_row(s: str) -> bool:
    """Похоже на строку таблицы: ≥2 пайпа и начинается/кончается на `|`.

    Строже, чем «есть пайп»: защищает от ложного срабатывания на прозе со
    случайным `|` (она почти никогда не обрамлена пайпами по краям)."""
    t = s.strip()
    return t.count("|") >= 2 and (t.startswith("|") or t.endswith("|"))


def segment_units(md: str):
    units = []
    lines = md.split("\n")
    i, n = 0, len(lines)
    buf = []

    def flush():
        text = " ".join(x.strip() for x in buf if x.strip())
        for s in sentenize(text):
            st = s.text.strip()
            if st:
                units.append(("sent", st))
        buf.clear()

    while i < n:
        ln = lines[i]

        # fenced code block → атомарный юнит
        if ln.strip().startswith("```"):
            blk = [ln]; i += 1
            while i < n and not lines[i].strip().startswith("```"):
                blk.append(lines[i]); i += 1
            if i < n:
                blk.append(lines[i]); i += 1
            flush(); units.append(("code", "\n".join(blk))); continue

        # markdown-таблица с разделителем (header + |---| + body)
        if "|" in ln and i + 1 < n and _is_sep(lines[i + 1]):
            flush()
            blk = [ln, lines[i + 1]]; i += 2
            while i < n and "|" in lines[i]:
                # FIX (баг B): встретили ещё одну строку-разделитель → значит
                # предыдущая добавленная строка была header'ом НОВОЙ таблицы.
                # Закрываем текущую таблицу и начинаем следующую — соседние
                # таблицы без пустой строки между ними больше не склеиваются.
                if _is_sep(lines[i]) and blk and not _is_sep(blk[-1]):
                    new_hdr = blk.pop()
                    units.append(("table", "\n".join(blk)))
                    blk = [new_hdr, lines[i]]; i += 1
                    continue
                blk.append(lines[i]); i += 1
            units.append(("table", "\n".join(blk))); continue

        # FIX (баг C): таблица БЕЗ строки-разделителя (≥2 подряд pipe-строк).
        # Без этого «кривой» markdown (из копипасты, без `|---|`) разъезжался
        # на прозовые предложения и мог быть порезан посреди таблицы.
        if _is_row(ln) and i + 1 < n and _is_row(lines[i + 1]):
            flush()
            blk = []
            while i < n and _is_row(lines[i]):
                blk.append(lines[i]); i += 1
            units.append(("table", "\n".join(blk))); continue

        if ln.strip() == "":
            flush(); i += 1; continue

        # markdown-заголовок → отдельный юнит
        if re.match(r"^#{1,6}\s", ln.strip()):
            flush(); units.append(("head", ln.strip())); i += 1; continue

        buf.append(ln); i += 1

    flush()
    return units
