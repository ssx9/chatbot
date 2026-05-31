from typing import Any

from pydantic_ai import RunContext

from deps import Deps

OPEN_METEO_FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'

WEATHER_CODE_MAP = {
    0: 'Clear sky',
    1: 'Mainly clear',
    2: 'Partly cloudy',
    3: 'Overcast',
    45: 'Fog',
    48: 'Depositing rime fog',
    51: 'Light drizzle',
    53: 'Moderate drizzle',
    55: 'Dense drizzle',
    56: 'Light freezing drizzle',
    57: 'Dense freezing drizzle',
    61: 'Slight rain',
    63: 'Moderate rain',
    65: 'Heavy rain',
    66: 'Light freezing rain',
    67: 'Heavy freezing rain',
    71: 'Slight snow fall',
    73: 'Moderate snow fall',
    75: 'Heavy snow fall',
    77: 'Snow grains',
    80: 'Slight rain showers',
    81: 'Moderate rain showers',
    82: 'Violent rain showers',
    85: 'Slight snow showers',
    86: 'Heavy snow showers',
    95: 'Thunderstorm',
    96: 'Thunderstorm with slight hail',
    99: 'Thunderstorm with heavy hail',
}


async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get current weather by latitude and longitude via Open-Meteo.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    r = await ctx.deps.client.get(
        OPEN_METEO_FORECAST_URL,
        params={
            'latitude': lat,
            'longitude': lng,
            'current': 'temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m',
            'timezone': 'auto',
        },
    )
    r.raise_for_status()

    payload = r.json()
    current = payload.get('current') or {}
    units = payload.get('current_units') or {}

    weather_code = current.get('weather_code')
    weather_description = WEATHER_CODE_MAP.get(weather_code, f'Unknown weather code ({weather_code})')

    return {
        'temperature': f"{current.get('temperature_2m')} {units.get('temperature_2m', '°C')}",
        'feels_like': f"{current.get('apparent_temperature')} {units.get('apparent_temperature', '°C')}",
        'humidity': f"{current.get('relative_humidity_2m')} {units.get('relative_humidity_2m', '%')}",
        'wind_speed': f"{current.get('wind_speed_10m')} {units.get('wind_speed_10m', 'km/h')}",
        'description': weather_description,
    }
