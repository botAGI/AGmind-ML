"""Честный тест-сет сегментации таблиц для context-aware сплиттера.

Проверяет САМ сегментатор (хост), от которого зависит целостность таблиц —
модель тут не участвует, тест герметичен (нужен только `razdel`).

Метрика на каждый кейс:
  • intact    — каждая строка таблицы целиком лежит в ОДНОМ table-юните
                (ни одна строка не утекла в прозу и не разорвана между юнитами);
  • isolated  — число table-юнитов == числу таблиц в документе
                (соседние таблицы не склеены в один юнит);
  • detected  — таблица распознана как table-юнит (а не разъехалась на прозу).

Запуск:  pytest tests/test_segmenter.py        — или —   python tests/test_segmenter.py
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))
from segmenter import segment_units  # noqa: E402


def _table_units(units):
    return [txt for typ, txt in units if typ == "table"]


def _sent_text(units):
    return "\n".join(txt for typ, txt in units if typ == "sent")


def assert_tables(md, tables, *, label):
    """tables — список таблиц, каждая = список её строк (точные подстроки).
    Проверяет detected + isolated + intact."""
    units = segment_units(md)
    tbl_units = _table_units(units)
    sents = _sent_text(units)

    # isolated: ровно столько table-юнитов, сколько таблиц
    assert len(tbl_units) == len(tables), (
        f"[{label}] ожидалось {len(tables)} table-юнитов, получено {len(tbl_units)} "
        f"(типы: {[t for t, _ in units]})")

    for ti, rows in enumerate(tables):
        # intact: все строки таблицы — внутри ОДНОГО table-юнита
        host = [u for u in tbl_units if all(r in u for r in rows)]
        assert len(host) == 1, (
            f"[{label}] таблица #{ti}: её строки не лежат целиком в одном "
            f"table-юните (нашлось {len(host)} подходящих)")
        # ни одна строка таблицы не утекла в прозу
        for r in rows:
            assert r not in sents, (
                f"[{label}] таблица #{ti}: строка {r!r} утекла в прозовый юнит")
    return units


# ---------------------------------------------------------------- кейсы

def test_single_clean_table():
    md = """Итоги квартала ниже.
| Регион | Выручка |
|---|---|
| Москва | 120 |
| Урал | 80 |
Совет утвердил бюджет."""
    units = assert_tables(md, [["| Москва | 120 |", "| Урал | 80 |"]], label="single")
    assert any(t == "sent" for t, _ in units)  # проза не пропала


def test_two_tables_separated_by_prose():
    """Кейс A — модель уже справлялась; сегментатор должен дать 2 юнита."""
    md = """Выручка по регионам.
| Регион | Выручка |
|---|---|
| Москва | 120 |
| Урал | 80 |
Затраты на персонал.
| Отдел | ФОТ |
|---|---|
| Продажи | 8.0 |
| IT | 4.5 |
Бюджет утверждён."""
    assert_tables(md, [["| Москва | 120 |"], ["| Продажи | 8.0 |"]], label="two-prose")


def test_adjacent_tables_no_blank_line():
    """БАГ B: две таблицы подряд без пустой строки — раньше склеивались в 1 юнит."""
    md = """Сравнение тарифов.
| Тариф | Цена |
|---|---|
| Старт | 300 |
| Про | 700 |
| Надбавка | Коэффициент |
|---|---|
| Север | 1.15 |
| Юг | 1.05 |"""
    assert_tables(
        md,
        [["| Старт | 300 |", "| Про | 700 |"], ["| Север | 1.15 |", "| Юг | 1.05 |"]],
        label="adjacent",
    )


def test_malformed_table_without_separator():
    """БАГ C: таблица без строки-разделителя |---| — раньше уезжала в прозу."""
    md = """Прайс на услуги ниже.
| Услуга | Цена |
| Аудит | 50000 |
| Внедрение | 120000 |
| Поддержка | 30000 |
Цены без НДС."""
    assert_tables(
        md,
        [["| Услуга | Цена |", "| Аудит | 50000 |", "| Поддержка | 30000 |"]],
        label="malformed",
    )


def test_big_table_stays_atomic():
    rows = "\n".join(f"| Товар-{k} | {k*100} | склад-{k%3} |" for k in range(40))
    md = f"""Складские остатки.
| Товар | Цена | Склад |
|---|---|---|
{rows}
Отчёт сформирован автоматически."""
    units = assert_tables(
        md, [[f"| Товар-{k} | {k*100} | склад-{k%3} |" for k in range(40)]], label="big"
    )
    assert len(_table_units(units)) == 1  # вся таблица — один юнит, не разорвана


def test_code_block_atomic_regression():
    md = """Пример конфигурации.
```yaml
key: value
nested:
  - a
  - b
```
Конец примера."""
    units = segment_units(md)
    assert sum(1 for t, _ in units if t == "code") == 1, "code-блок не атомарен"
    assert sum(1 for t, _ in units if t == "table") == 0, "code ошибочно принят за таблицу"


def test_prose_with_stray_pipe_not_a_table():
    """Защита от ложного срабатывания: проза со случайным `|` — не таблица."""
    md = "Это стоит 5 | 10 рублей в зависимости от объёма. Следующее предложение обычное."
    units = segment_units(md)
    assert sum(1 for t, _ in units if t == "table") == 0, "проза ошибочно принята за таблицу"


# ---------------------------------------------------------------- standalone runner
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1; print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {t.__name__}\n      {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
