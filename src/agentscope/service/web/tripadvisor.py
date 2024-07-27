# -*- coding: utf-8 -*-
"""TripAdvisor APIs for searching and retrieving location information."""

from typing import List, Any, Dict
from loguru import logger
import requests
from agentscope.service.service_response import ServiceResponse
from agentscope.service.service_status import ServiceExecStatus


def get_tripadvisor_location_photos(
    api_key: str,
    location_id: str,
    language: str = "en",
) -> ServiceResponse:
    """
    Retrieve photos for a specific location using the TripAdvisor API.

    Args:
        api_key (`str`):
            Your TripAdvisor API key.
        location_id (`str`):
            The ID of the location for which to retrieve photos.
        language (`str`, optional):
            The language for the response. Defaults to 'en'.

    Returns:
        `ServiceResponse`: A dictionary with two variables: `status` and
        `content`. The `status` variable is from the ServiceExecStatus enum,
        and `content` is the JSON response from TripAdvisor API or error
        information, which depends on the `status` variable.

        If successful, the `content` will be a dictionary with the following structure:
        {
            'photo_data': {
                'data': [
                    {
                        'id': int,
                        'is_blessed': bool,
                        'caption': str,
                        'published_date': str,
                        'images': {
                            'thumbnail': {'height': int, 'width': int, 'url': str},
                            'small': {'height': int, 'width': int, 'url': str},
                            'medium': {'height': int, 'width': int, 'url': str},
                            'large': {'height': int, 'width': int, 'url': str},
                            'original': {'height': int, 'width': int, 'url': str}
                        },
                        'album': str,
                        'source': {'name': str, 'localized_name': str},
                        'user': {'username': str}
                    },
                    ...
                ]
            }
        }
        Each item in the 'data' list represents a photo associated with the location.

    Example:
        .. code-block:: python

            result = get_tripadvisor_location_photos("your_api_key", "123456", "en")
            if result.status == ServiceExecStatus.SUCCESS:
                print(result.content)

    Example of successful `content`:
        {
            'photo_data': {
                'data': [
                    {
                        'id': 215321638,
                        'is_blessed': False,
                        'caption': '',
                        'published_date': '2016-09-04T20:40:14.284Z',
                        'images': {
                            'thumbnail': {'height': 50, 'width': 50, 'url': 'https://media-cdn.tripadvisor.com/media/photo-t/0c/d5/8c/26/photo0jpg.jpg'},
                            'small': {'height': 150, 'width': 150, 'url': 'https://media-cdn.tripadvisor.com/media/photo-l/0c/d5/8c/26/photo0jpg.jpg'},
                            'medium': {'height': 188, 'width': 250, 'url': 'https://media-cdn.tripadvisor.com/media/photo-f/0c/d5/8c/26/photo0jpg.jpg'},
                            'large': {'height': 413, 'width': 550, 'url': 'https://media-cdn.tripadvisor.com/media/photo-s/0c/d5/8c/26/photo0jpg.jpg'},
                            'original': {'height': 1920, 'width': 2560, 'url': 'https://media-cdn.tripadvisor.com/media/photo-c/2560x500/0c/d5/8c/26/photo0jpg.jpg'}
                        },
                        'album': 'Other',
                        'source': {'name': 'Traveler', 'localized_name': 'Traveler'},
                        'user': {'username': 'EvaFalleth'}
                    },
                    # ... more photo entries ...
                ]
            }
        }
    """

    url = (
        f"https://api.content.tripadvisor.com/api/v1/location/{location_id}/"
        f"photos?language={language}&key={api_key}"
    )
    headers = {
        "accept": "application/json",
    }

    logger.info(f"Requesting photos for location ID {location_id}")

    try:
        response = requests.get(url, headers=headers)
        logger.info(
            f"Received response with status code {response.status_code}",
        )

        if response.status_code == 200:
            logger.info("Successfully retrieved the photo")
            return ServiceResponse(
                status=ServiceExecStatus.SUCCESS,
                content=response.json(),
            )
        error_detail = (
            response.json()
            .get("error", {})
            .get("message", f"HTTP Error: {response.status_code}")
        )
        logger.error(f"Error in response: {error_detail}")
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content={"error": error_detail},
        )
    except Exception as e:
        logger.exception("Exception occurred while requesting location photos")
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content={"error": str(e)},
        )


def search_tripadvisor(
    api_key: str,
    query: str,
    language: str = "en",
) -> ServiceResponse:
    """
    Search for locations using the TripAdvisor API.

    Args:
        api_key (`str`):
            Your TripAdvisor API key.
        query (`str`):
            The search query.
        language (`str`, optional):
            The language for the response. Defaults to 'en'.

    Returns:
        `ServiceResponse`: A dictionary with two variables: `status` and
        `content`. The `status` variable is from the ServiceExecStatus enum,
        and `content` is the JSON response from TripAdvisor API or error
        information, which depends on the `status` variable.

        If successful, the `content` will be a dictionary with the following structure:
        {
            'data': [
                {
                    'location_id': str,
                    'name': str,
                    'address_obj': {
                        'street1': str,
                        'street2': str,
                        'city': str,
                        'state': str,
                        'country': str,
                        'postalcode': str,
                        'address_string': str
                    }
                },
                ...
            ]
        }
        Each item in the 'data' list represents a location matching the search query.

    Example:
        .. code-block:: python

            result = search_tripadvisor("your_api_key", "Socotra", "en")
            if result.status == ServiceExecStatus.SUCCESS:
                print(result.content)

    Example of successful `content`:
        {
            'data': [
                {
                    'location_id': '574818',
                    'name': 'Socotra Island',
                    'address_obj': {
                        'street2': '',
                        'city': 'Aden',
                        'country': 'Yemen',
                        'postalcode': '',
                        'address_string': 'Aden Yemen'
                    }
                },
                {
                    'location_id': '25395815',
                    'name': 'Tour Socotra',
                    'address_obj': {
                        'street1': '20th Street',
                        'city': 'Socotra Island',
                        'state': 'Socotra Island',
                        'country': 'Yemen',
                        'postalcode': '111',
                        'address_string': '20th Street, Socotra Island 111 Yemen'
                    }
                },
                # ... more results ...
            ]
        }
    """
    url = (
        f"https://api.content.tripadvisor.com/api/v1/location/search?"
        f"searchQuery={query}&language={language}&key={api_key}"
    )
    headers = {
        "accept": "application/json",
    }

    logger.info(f"Searching for locations with query '{query}'")

    try:
        response = requests.get(url, headers=headers)
        logger.info(
            f"Received response with status code {response.status_code}",
        )

        if response.status_code == 200:
            logger.info("Successfully retrieved search results")
            return ServiceResponse(
                status=ServiceExecStatus.SUCCESS,
                content=response.json(),
            )
        error_detail = (
            response.json()
            .get("error", {})
            .get("message", f"HTTP Error: {response.status_code}")
        )
        logger.error(f"Error in response: {error_detail}")
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content={"error": error_detail},
        )
    except Exception as e:
        logger.exception("Exception occurred while searching for locations")
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content={"error": str(e)},
        )


def get_tripadvisor_location_details(
    api_key: str,
    location_id: str,
    language: str = "en",
    currency: str = "USD",
) -> ServiceResponse:
    """
    Get detailed information about a specific location using the TripAdvisor API.

    Args:
        api_key (`str`):
            Your TripAdvisor API key.
        location_id (`str`):
            The unique identifier for the location.
        language (`str`, optional):
            The language for the response. Defaults to 'en'.
        currency (`str`, optional):
            The currency for pricing information. Defaults to 'USD'.

    Returns:
        `ServiceResponse`: A dictionary with two variables: `status` and
        `content`. The `status` variable is from the ServiceExecStatus enum,
        and `content` is the JSON response from TripAdvisor API or error
        information, which depends on the `status` variable.

        If successful, the `content` will be a dictionary with detailed information
        about the location, including name, address, ratings, reviews, and more.

    Example:
        .. code-block:: python

            result = get_tripadvisor_location_details("your_api_key", "574818", "en", "USD")
            if result.status == ServiceExecStatus.SUCCESS:
                print(result.content)

    Example of successful `content`:
        {
            'location_id': '574818',
            'name': 'Socotra Island',
            'web_url': 'https://www.tripadvisor.com/Attraction_Review-g298087-d574818-Reviews-Socotra_Island-Aden.html?m=66827',
            'address_obj': {
                'street2': '',
                'city': 'Aden',
                'country': 'Yemen',
                'postalcode': '',
                'address_string': 'Aden Yemen'
            },
            'ancestors': [
                {'level': 'City', 'name': 'Aden', 'location_id': '298087'},
                {'level': 'Country', 'name': 'Yemen', 'location_id': '294014'}
            ],
            'latitude': '12.46342',
            'longitude': '53.82374',
            'timezone': 'Asia/Aden',
            'write_review': 'https://www.tripadvisor.com/UserReview-g298087-d574818-Socotra_Island-Aden.html?m=66827',
            'ranking_data': {
                'geo_location_id': '298087',
                'ranking_string': '#1 of 7 things to do in Aden',
                'geo_location_name': 'Aden',
                'ranking_out_of': '7',
                'ranking': '1'
            },
            'rating': '5.0',
            'rating_image_url': 'https://www.tripadvisor.com/img/cdsi/img2/ratings/traveler/5.0-66827-5.svg',
            'num_reviews': '62',
            'review_rating_count': {'1': '1', '2': '0', '3': '1', '4': '1', '5': '59'},
            'photo_count': '342',
            'see_all_photos': 'https://www.tripadvisor.com/Attraction_Review-g298087-d574818-m66827-Reviews-Socotra_Island-Aden.html#photos',
            'category': {'name': 'attraction', 'localized_name': 'Attraction'},
            'subcategory': [
                {'name': 'nature_parks', 'localized_name': 'Nature & Parks'},
                {'name': 'attractions', 'localized_name': 'Attractions'}
            ],
            'groups': [
                {
                    'name': 'Nature & Parks',
                    'localized_name': 'Nature & Parks',
                    'categories': [{'name': 'Islands', 'localized_name': 'Islands'}]
                }
            ],
            'neighborhood_info': [],
            'trip_types': [
                {'name': 'business', 'localized_name': 'Business', 'value': '2'},
                {'name': 'couples', 'localized_name': 'Couples', 'value': '10'},
                {'name': 'solo', 'localized_name': 'Solo travel', 'value': '11'},
                {'name': 'family', 'localized_name': 'Family', 'value': '2'},
                {'name': 'friends', 'localized_name': 'Friends getaway', 'value': '22'}
            ],
            'awards': []
        }
    """
    url = (
        f"https://api.content.tripadvisor.com/api/v1/location/{location_id}/"
        f"details?language={language}&currency={currency}&key={api_key}"
    )
    headers = {
        "accept": "application/json",
    }

    logger.info(f"Requesting details for location ID {location_id}")

    try:
        response = requests.get(url, headers=headers)
        logger.info(
            f"Received response with status code {response.status_code}",
        )

        if response.status_code == 200:
            logger.info("Successfully retrieved location details")
            return ServiceResponse(
                status=ServiceExecStatus.SUCCESS,
                content=response.json(),
            )
        error_detail = (
            response.json()
            .get("error", {})
            .get("message", f"HTTP Error: {response.status_code}")
        )
        logger.error(f"Error in response: {error_detail}")
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content={"error": error_detail},
        )
    except Exception as e:
        logger.exception(
            "Exception occurred while requesting location details",
        )
        return ServiceResponse(
            status=ServiceExecStatus.ERROR,
            content={"error": str(e)},
        )
