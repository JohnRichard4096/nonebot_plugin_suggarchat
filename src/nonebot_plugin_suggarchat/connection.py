from nonebot import get_driver, logger
from nonebot.adapters import Bot

from .config import __KERNEL_VERSION__, config_manager

driver = get_driver()


@driver.on_bot_connect
async def onConnect(bot: Bot):
    logger.info(f"已连接 {bot.self_id} ，开始加载配置文件。")
    config_manager.load(bot.self_id)
    logger.info("配置文件加载完成。")
    logger.info(f"配置文件目录：{config_manager.config_dir}")
    logger.info(f"主要配置文件：{config_manager.toml_config}")
    logger.info(f"群聊记忆文件目录：{config_manager.group_memory}")
    logger.info(f"私聊记忆文件目录：{config_manager.private_memory}")
    logger.info(f"模型预设文件目录：{config_manager.custom_models_dir}")


@driver.on_startup
async def onEnable():
    logger.info(
        f"""
NONEBOT PLUGIN SUGGARCHAT
{__KERNEL_VERSION__}
"""
    )

    logger.info("Start successfully!Waitting for bot connection...")
