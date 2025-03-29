import json
import re
from datetime import datetime
from pathlib import Path

import chardet
import jieba
import nonebot
import pytz
from nonebot.adapters import Bot
from nonebot.adapters.onebot.v11 import (
    Event,
    GroupMessageEvent,
    Message,
    PokeNotifyEvent,
    PrivateMessageEvent,
)
from nonebot.log import logger

from .config import config_manager


def format_datetime_timestamp(time: int) -> str:
    now = datetime.fromtimestamp(time)

    # 格式化日期、星期和时间
    formatted_date = now.strftime("%Y-%m-%d")
    formatted_weekday = now.strftime("%A")
    formatted_time = now.strftime("%I:%M:%S %p")

    return f"[{formatted_date} {formatted_weekday} {formatted_time}]"


def hybrid_token_count(text: str, mode: str = "word") -> int:
    """
    混合中英文的 Token 计算方法（支持词、子词、字符模式）

    :param text: 输入文本
    :param mode: 统计模式，可选 'word'(词), 'bpe'(子词), 'char'(字符)
    :return: Token 数量
    """
    # 分离中英文部分（中文按结巴分词，英文按空格/标点分割）
    chinese_parts = re.findall(r"[\u4e00-\u9fff]+", text)
    non_chinese_parts = re.split(r"([\u4e00-\u9fff]+)", text)

    tokens = []

    # 处理中文部分（精准分词）
    for part in chinese_parts:
        tokens.extend(list(jieba.cut(part, cut_all=False)))  # 精准模式

    # 处理非中文部分（按空格和标点分割）
    for part in non_chinese_parts:
        if not part.strip() or part in chinese_parts:
            continue
        # 按正则匹配英文单词、数字、标点
        if mode == "word":
            tokens.extend(re.findall(r"\b\w+\b|\S", part))
        elif mode == "char":
            tokens.extend(list(part))
        elif mode == "bpe":
            # 简易BPE处理（示例：按2-gram拆分）
            tokens.extend([part[i : i + 2] for i in range(0, len(part), 2)])
        else:
            raise ValueError("Invalid tokens-counting mode")
    return len(tokens)


def split_message_into_chats(text):
    sentence_delimiters = re.compile(
        r'(?<=[a-zA-Z])\.|([。！？!?~]+)[”"’\']*',  # 关键修改：仅匹配字母后的英文句号
        re.UNICODE,
    )

    sentences = []
    start = 0
    for match in sentence_delimiters.finditer(text):
        end = match.end()
        if sentence := text[start:end].strip():
            sentences.append(sentence)
        start = end

    if start < len(text):
        if remaining := text[start:].strip():
            sentences.append(remaining)

    return sentences


def convert_to_utf8(file_path) -> bool:
    file_path = str(file_path)
    # 检测文件编码

    with open(file_path, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result["encoding"]
    if encoding is None:
        try:
            with open(file_path) as f:
                contents = f.read()
                if contents.strip() == "":
                    return True
        except Exception:
            logger.warning(f"无法读取文件{file_path}")
            return False
        logger.warning(f"无法检测到编码{file_path}")
        return False

    # 读取原文件并写入UTF-8编码的文件
    with open(file_path, encoding=encoding) as file:
        content = file.read()

    # 以UTF-8编码重新写入文件
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)
    return True


async def synthesize_message(message: Message, bot: Bot) -> str:
    content = ""
    for segment in message:
        if segment.type == "text":
            content = content + segment.data["text"]

        elif segment.type == "at":
            content += f"\\（at: @{segment.data.get('name')}(QQ:{segment.data['qq']}))"
        elif segment.type == "forward":
            forward = await bot.get_forward_msg(id=segment.data["id"])
            logger.debug(forward)
            content += (
                " \\（合并转发\n" + await synthesize_forward_message(forward) + "）\\\n"
            )
    return content


def get_memory_data(event: Event) -> dict:
    logger.debug(f"获取{event.get_type()} {event.get_session_id()} 的记忆数据")
    """
    根据消息事件获取记忆数据，如果用户或群组的记忆数据不存在，则创建初始数据结构

    参数:
    event: MessageEvent - 消息事件，可以是私聊消息事件或群聊消息事件，通过事件解析获取用户或群组ID

    返回:
    dict - 用户或群组的记忆数据字典
    """
    private_memory = config_manager.private_memory
    group_memory = config_manager.group_memory
    # 检查私聊记忆目录是否存在，如果不存在则创建
    if not Path(private_memory).exists() or not Path(private_memory).is_dir():
        Path.mkdir(private_memory)

    # 检查群聊记忆目录是否存在，如果不存在则创建
    if not Path(group_memory).exists() or not Path(group_memory).is_dir():
        Path.mkdir(group_memory)

    # 根据事件类型判断是私聊还是群聊
    if (
        (not isinstance(event, PrivateMessageEvent)
        and not isinstance(event, GroupMessageEvent)
        and isinstance(event, PokeNotifyEvent)
        and event.group_id)
        or (not isinstance(event, PrivateMessageEvent)
        and isinstance(event, GroupMessageEvent))
    ):
        group_id = event.group_id
        conf_path = Path(group_memory / f"{group_id}.json")
        if not conf_path.exists():
            with open(str(conf_path), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "id": group_id,
                        "enable": True,
                        "memory": {"messages": []},
                        "full": False,
                    },
                    f,
                    ensure_ascii=True,
                    indent=0,
                )
    elif (
        (not isinstance(event, PrivateMessageEvent)
        and isinstance(event, PokeNotifyEvent))
        or isinstance(event, PrivateMessageEvent)
    ):
        user_id = event.user_id
        conf_path = Path(private_memory / f"{user_id}.json")
        if not conf_path.exists():
            with open(str(conf_path), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "id": user_id,
                        "enable": True,
                        "memory": {"messages": []},
                        "full": False,
                    },
                    f,
                    ensure_ascii=True,
                    indent=0,
                )
    convert_to_utf8(conf_path)
    # 读取并返回记忆数据
    with open(str(conf_path), encoding="utf-8") as f:
        conf = json.load(f)
        logger.debug(f"读取到记忆数据{conf}")
        return conf


def write_memory_data(event: Event, data: dict) -> None:
    logger.debug(f"写入记忆数据{data}")
    logger.debug(f"事件：{type(event)}")
    """
    根据事件类型将数据写入到特定的记忆数据文件中。

    该函数根据传入的事件类型（群组消息事件或用户消息事件），将相应的数据以JSON格式写入到对应的文件中。
    对于群组消息事件，数据被写入到以群组ID命名的文件中；对于用户消息事件，数据被写入到以用户ID命名的文件中。

    参数:
    - event: MessageEvent类型，表示一个消息事件，可以是群组消息事件或用户消息事件。
    - data: dict类型，要写入的数据，以字典形式提供。

    返回值:
    无返回值。
    """
    group_memory = config_manager.group_memory
    private_memory = config_manager.private_memory

    # 判断事件是否为群组消息事件
    if isinstance(event, GroupMessageEvent):
        # 获取群组ID，并根据群组ID构造配置文件路径
        group_id = event.group_id
        conf_path = Path(group_memory / f"{group_id}.json")
    elif isinstance(event, PrivateMessageEvent):
        # 获取用户ID，并根据用户ID构造配置文件路径
        user_id = event.user_id
        conf_path = Path(private_memory / f"{user_id}.json")
    elif isinstance(event, PokeNotifyEvent):
        if event.group_id:
            group_id = event.group_id
            conf_path = Path(group_memory / f"{group_id}.json")
            if not conf_path.exists():
                with open(str(conf_path), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "id": group_id,
                            "enable": True,
                            "memory": {"messages": []},
                            "full": False,
                        },
                        f,
                        ensure_ascii=True,
                        indent=0,
                    )
        else:
            user_id = event.user_id
            conf_path = Path(private_memory / f"{user_id}.json")
            if not conf_path.exists():
                with open(str(conf_path), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "id": user_id,
                            "enable": True,
                            "memory": {"messages": []},
                            "full": False,
                        },
                        f,
                        ensure_ascii=True,
                        indent=0,
                    )
    # 打开配置文件路径对应的文件，以写入模式，并确保文件以UTF-8编码
    with open(str(conf_path), "w", encoding="utf-8") as f:
        # 将数据写入到文件中，确保ASCII字符以外的字符也能被正确处理
        json.dump(data, f, ensure_ascii=True)


async def get_friend_info(qq_number: int) -> str:
    bot = nonebot.get_bot()  # 假设只有一个Bot实例运行
    friend_list = await bot.get_friend_list()

    return next(
        (
            friend["nickname"]
            for friend in friend_list
            if friend["user_id"] == qq_number
        ),
        "",
    )


async def get_friend_qq_list():
    bot = nonebot.get_bot()
    friend_list = await bot.get_friend_list()
    return [friend["user_id"] for friend in friend_list]


def split_list(lst: list, threshold: int) -> list:
    """
    将列表分割成多个子列表，每个子列表的最大长度不超过threshold。

    :param lst: 原始列表
    :param threshold: 子列表的最大长度
    :return: 分割后的子列表列表
    """
    if len(lst) <= threshold:
        return [lst]

    return [lst[i : i + threshold] for i in range(0, len(lst), threshold)]


async def get_group_member_qq_numbers(group_id: int) -> list[int]:
    """
    获取指定群组的所有成员QQ号列表

    :param group_id: 群组ID
    :return: 成员QQ号列表
    """
    bot = nonebot.get_bot()  # 获取当前机器人实例
    member_list = await bot.get_group_member_list(group_id=group_id)

    return [member["user_id"] for member in member_list]


async def is_same_day(timestamp1: int, timestamp2: int) -> bool:
    # 将时间戳转换为datetime对象，并只保留日期部分
    date1 = datetime.fromtimestamp(timestamp1).date()
    date2 = datetime.fromtimestamp(timestamp2).date()

    # 比较两个日期是否相同
    return date1 == date2


async def synthesize_forward_message(forward_msg: dict) -> str:
    forw_msg = forward_msg
    # 初始化最终字符串
    result = ""

    # forward_msg 是一个包含多个消息段的字典+列表
    for segment in forw_msg["messages"]:
        nickname = segment["sender"]["nickname"]
        qq = segment["sender"]["user_id"]
        time = f"[{datetime.fromtimestamp(segment['time']).strftime('%Y-%m-%d %I:%M:%S %p')}]"
        result += f"{time}[{nickname}({qq})]说："
        for segments in segment["content"]:
            segments_type = segments["type"]
            if segments_type == "text":
                result += f"{segments['data']['text']}"

            elif segments_type == "at":
                result += f" [@{segments['data']['qq']}]"

        result += "\n"

    return result


def get_current_datetime_timestamp():
    # 获取当前 UTC 时间
    utc_time = datetime.now(pytz.utc)

    # 转换为+8时区的时间
    asia_shanghai = pytz.timezone("Asia/Shanghai")
    now = utc_time.astimezone(asia_shanghai)

    # 格式化日期、星期和时间（24小时制）
    formatted_date = now.strftime("%Y-%m-%d")  # 日期保持原格式[1](@ref)
    formatted_weekday = now.strftime("%A")  # 星期保持完整名称[9](@ref)
    formatted_time = now.strftime("%H:%M:%S")  # 关键修改点：%H 表示24小时制[2,8](@ref)

    return f"[{formatted_date} {formatted_weekday} {formatted_time}]"
