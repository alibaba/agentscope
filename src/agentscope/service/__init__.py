# -*- coding: utf-8 -*-
""" Import all service-related modules in the package."""
from loguru import logger

from .execute_code.exec_python import execute_python_code
from .execute_code.exec_shell import execute_shell_command
from .execute_code.exec_notebook import NoteBookExecutor
from .file.common import (
    create_file,
    delete_file,
    move_file,
    create_directory,
    delete_directory,
    move_directory,
    list_directory_content,
    get_current_directory,
)
from .file.text import read_text_file, write_text_file
from .file.json import read_json_file, write_json_file
from .sql_query.mysql import query_mysql
from .sql_query.sqlite import query_sqlite
from .sql_query.mongodb import query_mongodb
from .web.search import bing_search, google_search
from .web.arxiv import arxiv_search
from .web.tripadvisor import (
    tripadvisor_search_location_photos,
    tripadvisor_search,
    tripadvisor_search_location_details,
)
from .web.dblp import (
    dblp_search_publications,
    dblp_search_authors,
    dblp_search_venues,
)
from .multi_modality.dashscope_services import (
    dashscope_image_to_text,
    dashscope_text_to_image,
    dashscope_text_to_audio,
)
from .multi_modality.openai_services import (
    openai_audio_to_text,
    openai_text_to_audio,
    openai_text_to_image,
    openai_image_to_text,
    openai_edit_image,
    openai_create_image_variation,
)
from .multi_modality.stablediffusion_services import sd_text_to_image

from .service_response import ServiceResponse
from .service_toolkit import ServiceToolkit
from .service_toolkit import ServiceFactory
from .retrieval.similarity import cos_sim
from .text_processing.summarization import summarization
from .retrieval.retrieval_from_list import retrieve_from_list
from .service_status import ServiceExecStatus
from .web.web_digest import digest_webpage, load_web, parse_html_to_text
from .web.download import download_from_url

from .web.wikipedia import (
    wikipedia_search,
    wikipedia_search_categories,
)

from .browser.web_browser import WebBrowser, WebElementInfo


def get_help() -> None:
    """Get help message."""

    help_msg = "\n - ".join(
        ["The following services are available:"] + __all__[4:],
    )
    logger.info(help_msg)


__all__ = [
    "ServiceResponse",
    "ServiceExecStatus",
    "ServiceToolkit",
    "get_help",
    "execute_python_code",
    "execute_shell_command",
    "create_file",
    "delete_file",
    "move_file",
    "create_directory",
    "delete_directory",
    "move_directory",
    "list_directory_content",
    "get_current_directory",
    "read_text_file",
    "write_text_file",
    "read_json_file",
    "write_json_file",
    "bing_search",
    "google_search",
    "arxiv_search",
    "wikipedia_search",
    "wikipedia_search_categories",
    "query_mysql",
    "query_sqlite",
    "query_mongodb",
    "cos_sim",
    "summarization",
    "retrieve_from_list",
    "digest_webpage",
    "load_web",
    "parse_html_to_text",
    "download_from_url",
    "dblp_search_publications",
    "dblp_search_authors",
    "dblp_search_venues",
    "NoteBookExecutor",
    "dashscope_image_to_text",
    "dashscope_text_to_image",
    "dashscope_text_to_audio",
    "openai_audio_to_text",
    "openai_text_to_audio",
    "openai_text_to_image",
    "openai_image_to_text",
    "openai_edit_image",
    "openai_create_image_variation",
    "sd_text_to_image",
    "tripadvisor_search",
    "tripadvisor_search_location_photos",
    "tripadvisor_search_location_details",
    "WebBrowser",
    "WebElementInfo",
    # to be deprecated
    "ServiceFactory",
]
