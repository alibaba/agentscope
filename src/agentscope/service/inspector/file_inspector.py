# -*- coding: utf-8 -*-
"""
The ServiceFunction utils for inspecting files.
Convert different file formats to LLM understandable text.
"""
from io import StringIO
import os
from typing import Optional, Union, Dict, Callable

import mammoth
import puremagic
import markdownify
from bs4 import BeautifulSoup
import pandas as pd
import pdfminer
import pdfminer.high_level
from agentscope.service import ServiceResponse, ServiceExecStatus


def _guess_ext_magic(file_path: str) -> Union[str, None]:
    """
    Use puremagic to guess a file's extension based on the first few bytes.
    """
    guesses = puremagic.magic_file(file_path)
    if len(guesses) > 0:
        ext = guesses[0].extension.strip()
        if len(ext) > 0:
            return ext

    return None


# currently supported file types
# plain_text, docx, xlsx, pdf, html, py, txt
# TODO support more types: image, audio, etc.
def inspect_file_as_text(file_path: str) -> ServiceResponse:
    """Inspect common-types files to markdown style text,
    so the llm can inspect such files.
    Currently support '.docx', '.xlsx', '.pdf', '.txt', '.py', '.html', etc.
    Args:
        file_path (str):
            the path of the file to inspect
    """
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if not ext:
            ext = _guess_ext_magic(file_path)
            if ext:
                ext = "." + ext.lower()

        inspect_functions: Dict[str, Callable[[str], ServiceResponse]] = {
            ".docx": inspect_docx_as_text,
            ".xlsx": inspect_xlsx_as_text,
            ".pdf": inspect_pdf_as_text,
            ".html": inspect_html_as_text,
            ".txt": inspect_raw_local_file,
            ".py": inspect_raw_local_file,
        }

        if ext in inspect_functions:
            return inspect_functions[ext](file_path)
        else:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content=f"Unsupported file extension: {ext}",
            )
    except FileNotFoundError as e:
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=str(e),
        )
    except Exception as e:
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content=f"An error occurred: {str(e)}",
        )


def parse_html_to_markdown(html_content: str) -> ServiceResponse:
    """Parse the html content to markdown format.
    Args:
        html_content (str):
            the html content to be parsed
    """
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        # Remove javascript and style blocks that may be too messy
        for block in soup(["script", "style"]):
            block.extract()
        # if there are main content, find the main content only
        body_elm = soup.find("body")
        content_text = ""
        if body_elm:
            content_text = markdownify.MarkdownConverter().convert_soup(
                body_elm,
            )
        else:
            content_text = markdownify.MarkdownConverter().convert_soup(soup)
        content_title = ""
        if soup.title:
            content_title = soup.title.string + "\n"
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=content_title + content_text,
        )
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR, content=str(e))


def inspect_docx_as_text(file_path: str) -> ServiceResponse:
    """Inspect the text content in the docx file.
    Args:
        file_path (str):
            the path of the docx file
    """
    try:
        with open(file_path, "rb") as docx_file:
            result = mammoth.convert_to_markdown(docx_file)
        markdown = result.value
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=markdown,
        )
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR, content=str(e))


def inspect_html_as_text(file_path: str) -> ServiceResponse:
    """Inspect the text content in the html file.
    Args:
        file_path (str):
            the path of the html file
    """
    try:
        with open(file_path, "rt") as file:  # pylint: disable=W1514
            parsed = parse_html_to_markdown(file.read())
        return parsed
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR, content=str(e))


def inspect_xlsx_as_text(file_path: str) -> ServiceResponse:
    """Inspect the content in the xlsx file.
    Args:
        file_path (str):
            the path of the xlsx file
    """
    try:
        sheets = pd.read_excel(file_path)
        md_buffer = StringIO()
        for sheet_name, sheet_data in sheets.items():
            md_buffer.write(f"## {sheet_name}\n")
            html_content = sheet_data.to_html(index=False)
            md_buffer.write(
                parse_html_to_markdown(html_content).content.strip(),
            )
            md_buffer.write("\n\n")
        md_content = md_buffer.getvalue().strip()
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=md_content,
        )
    except FileNotFoundError as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR, content=str(e))


def inspect_pdf_as_text(file_path: str) -> ServiceResponse:
    """Inspect the text content in the pdf file.
    Args:
        file_path (str):
            the path of the pdf file
    """
    # TODO we could consider using pdf conversion repos such as
    # https://github.com/VikParuchuri/marker for better performance.
    # However, such module would require
    # heavy local computations and dependencies.
    try:
        pdf_content = pdfminer.high_level.extract_text(file_path)
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=pdf_content,
        )
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR, content=str(e))


def inspect_raw_local_file(
    file_path: str,
    set_nu: Optional[bool] = True,
) -> ServiceResponse:
    """Inspect the content in the local file.
    Useful for '.py' and '.txt' files.

    Args:
        file_path (str):
            the path of the file
        set_nu (bool, optional):
            whether to show the line number in given the content.
            Defaults to True.
    """
    try:
        with open(file_path, "rt") as file:  # pylint: disable=W1514
            content = file.readlines()
            if set_nu:
                content = [f"{i+1}: {line}" for i, line in enumerate(content)]
            text_content = "".join(content)
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=text_content,
        )
    except Exception as e:
        return ServiceResponse(status=ServiceExecStatus.ERROR, content=str(e))
