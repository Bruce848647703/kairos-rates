"""日计数约定与日期工具。"""
from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Union

DateLike = Union[str, date, datetime]

CONVENTIONS = ("ACT/360", "ACT/365F", "30/360", "ACT/ACT")


def to_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"无法解析日期：{value!r}")


def add_months(anchor: date, months: int) -> date:
    """按月平移日期（自动处理月末）。"""
    total = anchor.month - 1 + int(months)
    year = anchor.year + total // 12
    month = total % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def add_years(anchor: date, years: int) -> date:
    return add_months(anchor, 12 * int(years))


def year_fraction(start: DateLike, end: DateLike, convention: str = "ACT/365F") -> float:
    """两个日期之间的年化期限。"""
    d0, d1 = to_date(start), to_date(end)
    if d1 < d0:
        raise ValueError(f"end ({d1}) 早于 start ({d0})")
    days = (d1 - d0).days
    convention = str(convention).upper()

    if convention == "ACT/360":
        return days / 360.0
    if convention in ("ACT/365F", "ACT/365", "ACT/365.25"):
        return days / 365.0
    if convention == "30/360":
        d0_day = min(d0.day, 30)
        d1_day = d1.day
        if d0_day == 30 and d1_day == 31:
            d1_day = 30
        return (360 * (d1.year - d0.year) + 30 * (d1.month - d0.month)
                + (d1_day - d0_day)) / 360.0
    if convention == "ACT/ACT":
        if d0.year == d1.year:
            return days / (366.0 if calendar.isleap(d0.year) else 365.0)
        total = 0.0
        cursor = d0
        while cursor.year < d1.year:
            year_end = date(cursor.year + 1, 1, 1)
            total += (year_end - cursor).days / (366.0 if calendar.isleap(cursor.year) else 365.0)
            cursor = year_end
        total += (d1 - cursor).days / (366.0 if calendar.isleap(d1.year) else 365.0)
        return total
    raise ValueError(f"未知日计数约定 {convention!r}，可用：{CONVENTIONS}")


def schedule(start: DateLike, end: DateLike, frequency: int = 2) -> list:
    """生成付息日列表（含起止），按频率从 end 向前回推。"""
    d0, d1 = to_date(start), to_date(end)
    frequency = int(frequency)
    if frequency < 1:
        raise ValueError("frequency 必须 >= 1")
    months = 12 // frequency
    if months * frequency != 12:
        raise ValueError("frequency 必须是 1/2/3/4/6/12")
    dates = [d1]
    cursor = d1
    while True:
        cursor = add_months(cursor, -months)
        if cursor <= d0:
            break
        dates.append(cursor)
    dates.append(d0)
    return sorted(dates)


__all__ = [
    "CONVENTIONS",
    "DateLike",
    "add_months",
    "add_years",
    "schedule",
    "to_date",
    "year_fraction",
]