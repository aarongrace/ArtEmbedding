import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from pydantic_settings import BaseSettings, SettingsConfigDict
from routers.profile import Profile, ensure_guest_profile_exists
from routers.colony import Colony, ensure_guest_colony_exists, ensure_all_colonies_valid
from routers.clan import Clan
from routers.trades import Trade
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

class MyConfig(BaseSettings):
    connect_string: str
    # hardcoding the env file path to the root because it keeps giving errors
    env_file: Path = str(Path(__file__).parent.parent / ".env")
    model_config = SettingsConfigDict(env_file=env_file)

async def initialize_database():
    setting = MyConfig()
    client = AsyncIOMotorClient(setting.connect_string)
    await init_beanie(database=client["clash_of_colonies"], document_models=[Profile, Colony, Clan, Trade])

    await ensure_all_colonies_valid()
    await ensure_guest_profile_exists()
    await ensure_guest_colony_exists(reinitialize=True)
    logger.info("Database initialization completed successfully.")


async def main():
    await initialize_database()


if __name__ == "__main__":
    asyncio.run(main())