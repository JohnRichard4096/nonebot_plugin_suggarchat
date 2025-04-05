from collections.abc import Callable

from nonebot import logger

from .config import Config as Conf
from .config import ConfigManager, config_manager
from .suggar import chat_manager, get_chat, protocols_adapters, send_to_admin

Config: ConfigManager = config_manager


class Adapter:
    """用于处理Adapter注册的类"""

    def __init__(self):
        """
        初始化 Adapter 类的新实例。
        """
        pass

    def register_adapter(self, func: Callable, protocol: str):
        """
        注册一个适配器。
        """
        if protocol in protocols_adapters:
            raise ValueError("协议适配器已存在")
        protocols_adapters[protocol] = func

    def get_adapter(self, protocol: str) -> Callable:
        """
        获取适配器方法。
        """
        if protocol not in protocols_adapters:
            raise ValueError("协议适配器不存在")
        return protocols_adapters[protocol]

    def get_adapters(self) -> dict:
        """
        获取适配器方法。
        """
        return protocols_adapters

    @property
    def adapters(self) -> dict:
        """
        获取适配器方法。
        """
        return protocols_adapters


class Menu:
    """
    Menu 类用于通过注册菜单项来构建菜单。
    """

    def __init__(self):
        """
        初始化 Menu 类的新实例。
        """
        pass

    def reg_menu(self, cmd_name: str, describe: str, args: str = "") -> "Menu":
        """
        注册一个新的菜单项。

        参数:
        - cmd_name (str): 菜单项的命令名称。
        - describe (str): 菜单项的描述。

        返回:
        - Menu: 返回 Menu 类的实例，支持方法链式调用。
        """
        chat_manager.menu_msg += f"/{cmd_name} {args} 说明：{describe} \n"
        return self

    @property
    def menu(self) -> str:
        """
        获取菜单项。
        """
        return chat_manager.menu_msg


class Admin:
    """
    管理员管理类，负责处理与管理员相关的操作，如发送消息、错误处理和管理员权限管理。
    """

    config: Conf

    def __init__(self):
        """
        构造函数
        """
        self.config = config_manager.config

    async def send_with(self, msg: str) -> "Admin":
        """
        异步发送消息给管理员。

        参数:
        - msg (str): 要发送的消息内容。

        返回:
        - Admin: 返回Admin实例，支持链式调用。
        """
        await send_to_admin(msg)
        return self

    async def send_error(self, msg: str) -> "Admin":
        """
        异步发送错误消息给管理员，并记录错误日志。

        参数:
        - msg (str): 要发送的错误消息内容。

        返回:
        - Admin: 返回Admin实例，支持链式调用。
        """
        logger.error(msg)
        await send_to_admin(msg)
        return self

    def is_admin(self, user_id: str) -> bool:
        """
        检查用户是否是管理员。

        参数:
        - user_id (str): 用户ID。

        返回:
        - bool: 用户是否是管理员。
        """
        return user_id in self.config.admins

    def add_admin(self, user_id: int) -> "Admin":
        """
        添加新的管理员用户ID到配置中。

        参数:
        - user_id (int): 要添加的用户ID。

        返回:
        - Admin: 返回Admin实例，支持链式调用。
        """
        self.config.admins.append(user_id)
        return self._save_config_to_toml()

    def set_admin_group(self, group_id: int) -> "Admin":
        """
        设置管理员组ID。

        参数:
        - group_id (int): 管理员组ID。

        返回:
        - Admin: 返回Admin实例，支持链式调用。
        """
        self.config.admin_group = group_id
        return self._save_config_to_toml()

    def _save_config_to_toml(self):
        self.config.save_to_toml(config_manager.toml_config)
        self.config = config_manager.config
        return self


class Chat:
    """
    Chat 类用于处理与LLM相关操作，如获取消息。
    """

    config: Conf

    def __init__(self):
        """
        构造函数
        """
        self.config = config_manager.config

    async def get_msg(self, prompt: str, message: list):
        """
        获取LLM响应

        :param prompt[str]: 提示词
        :param message[list]: 消息列表
        """

        message.insert(0, {"role": "assistant", "content": prompt})
        return await self.get_msg_on_list(message)

    async def get_msg_on_list(self, message: list):
        """
        获取LLM响应

        :param message[list]: 消息列表
        """

        return await get_chat(messages=message)
