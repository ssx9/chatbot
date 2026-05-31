from pydantic import BaseModel
from pydantic_ai import RunContext

from deps import Deps

OPEN_METEO_GEOCODING_URL = 'https://geocoding-api.open-meteo.com/v1/search'


class LatLng(BaseModel):
    lat: float
    lng: float


def _contains_chinese(text: str) -> bool:
    return any('一' <= ch <= '鿿' for ch in text)


async def _search_lat_lng(ctx: RunContext[Deps], location_description: str, language: str | None):
    params = {
        'name': location_description,
        'count': 1,
        'format': 'json',
    }
    if language:
        params['language'] = language

    r = await ctx.deps.client.get(OPEN_METEO_GEOCODING_URL, params=params)
    r.raise_for_status()
    payload = r.json()
    return payload.get('results') or []


async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    query = location_description.strip()
    if not query:
        raise ValueError('location_description cannot be empty')

    preferred_languages = ['zh', 'en'] if _contains_chinese(query) else ['en', 'zh']

    for language in preferred_languages:
        results = await _search_lat_lng(ctx, query, language)
        if results:
            best_match = results[0]
            return LatLng(lat=best_match['latitude'], lng=best_match['longitude'])

    results = await _search_lat_lng(ctx, query, None)
    if results:
        best_match = results[0]
        return LatLng(lat=best_match['latitude'], lng=best_match['longitude'])

    raise ValueError(f'Location not found: {location_description}')
