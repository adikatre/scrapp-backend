"""
The Battery Network — live locator API scraper (All Southern California)
=========================================================================

Endpoint discovered via browser DevTools on https://batterynetwork.org/locator/
This calls the same Azure API Management backend the site's own JS widget
calls, so it returns live, authoritative data straight from the source.

Base call shape (as observed):
  GET https://apim-techservices-integrations-prod-us.azure-api.net/Locator/locations_within_range/{lat}/{lng}/{radius_meters}
  Header: Ocp-Apim-Subscription-Key: <key>

SETUP
-----
    pip install requests
    export BATTERYNETWORK_APIM_KEY="ee294a0c63854ef0afcce0f8466dd583"

STEP 1 — PROBE THE RESPONSE SHAPE
------------------------------------
    python sd_county_live_locator.py --probe

STEP 2 — FULL SOUTHERN CALIFORNIA SWEEP
------------------------------------------
    python sd_county_live_locator.py

Covers all 10 Southern California counties:
  Los Angeles, San Diego, Orange, Riverside, San Bernardino,
  Ventura, Santa Barbara, Kern, San Luis Obispo, Imperial

Tiles query points across each county's bounding box with overlapping
30km-radius circles, dedupes by account number, filters by county
city lists + bounding-box fallback, and writes a combined CSV.
"""

import argparse
import csv
import json
import math
import os
import time

import requests

BASE_URL = (
    "https://apim-techservices-integrations-prod-us.azure-api.net"
    "/Locator/locations_within_range"
)

# Prefer an environment variable over hardcoding the key in source.
SUBSCRIPTION_KEY = os.environ.get(
    "BATTERYNETWORK_APIM_KEY", "ee294a0c63854ef0afcce0f8466dd583"
)

HEADERS = {
    "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY,
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; SoCalRecyclingResearch/1.0)",
}

RADIUS_METERS = 30000  # matches the sample call you captured
OVERLAP_FACTOR = 1.4   # >1.0 so adjacent circles overlap, no coverage gaps
REQUEST_DELAY_SECONDS = 0.5

# ---------------------------------------------------------------------------
# County configurations: bounding box (lat_min, lat_max, lng_min, lng_max)
# + known city/community names (lowercase) for filtering border-bleed results.
#
# City lists include all incorporated cities plus major unincorporated
# communities / CDPs that are likely to appear in business addresses.
# If a city name is NOT in the set, we fall back to a bounding-box check,
# so missing an obscure community is not catastrophic.
# ---------------------------------------------------------------------------
COUNTIES = {
    # --- Existing 3 counties (verified working) ---
    "San Diego": {
        "bbox": (32.53, 33.51, -117.60, -116.08),
        "cities": {
            "san diego", "chula vista", "encinitas", "escondido", "oceanside",
            "carlsbad", "poway", "el cajon", "la mesa", "national city",
            "santee", "vista", "san marcos", "lemon grove", "coronado",
            "del mar", "imperial beach", "solana beach", "spring valley",
            "ramona", "alpine", "bonita", "jamul", "lakeside", "valley center",
            "fallbrook", "julian", "borrego springs", "rancho santa fe",
            "descanso", "pine valley", "potrero", "campo", "boulevard",
            "pauma valley", "santa ysabel", "warner springs", "bonsall",
            "cardiff", "cardiff by the sea", "leucadia", "olivenhain",
            "rancho bernardo", "rancho penasquitos", "scripps ranch",
            "mira mesa", "clairemont", "kearny mesa", "tierrasanta",
            "san ysidro", "otay mesa", "paradise hills", "encanto",
            "barrio logan", "logan heights", "ocean beach", "pacific beach",
            "mission beach", "la jolla", "university city", "sorrento valley",
            "carmel valley", "torrey pines", "del cerro", "college area",
            "north park", "hillcrest", "mission hills", "normal heights",
            "city heights", "mid-city", "linda vista", "serra mesa",
            "bay park", "bay ho", "nestor", "palm city", "tecate",
            "jacumba", "mount laguna", "harbison canyon", "crest",
            "dehesa", "flinn springs", "granite hills", "bostonia",
            "winter gardens", "casa de oro-mount helix", "casa de oro",
            "mount helix", "la presa", "lincoln acres", "rancho san diego",
        },
    },
    "Imperial": {
        "bbox": (32.62, 33.43, -116.11, -114.46),
        "cities": {
            "el centro", "calexico", "brawley", "imperial", "holtville",
            "calipatria", "westmorland", "seeley", "heber", "niland",
            "bombay beach", "ocotillo", "plaster city", "winterhaven",
            "palo verde", "salton city", "desert shores", "north shore",
            "glamis", "andrade", "felicity", "mount signal",
        },
    },
    "Riverside": {
        "bbox": (33.43, 34.08, -117.67, -114.51),
        "cities": {
            "riverside", "moreno valley", "corona", "temecula", "murrieta",
            "menifee", "indio", "hemet", "perris", "lake elsinore",
            "palm desert", "palm springs", "cathedral city", "san jacinto",
            "beaumont", "coachella", "la quinta", "wildomar", "eastvale",
            "jurupa valley", "desert hot springs", "banning", "norco",
            "calimesa", "indian wells", "rancho mirage", "canyon lake",
            "thousand palms", "mecca", "thermal", "oasis", "anza",
            "aguanga", "idyllwild", "idyllwild-pine cove", "winchester",
            "nuevo", "homeland", "romoland", "sun city", "lakeland village",
            "temescal valley", "woodcrest", "glen avon", "mead valley",
            "bermuda dunes", "sky valley", "north palm springs",
            "cabazon", "whitewater", "blythe", "ripley",
            "lake riverside", "french valley", "good hope",
            "highgrove", "home gardens", "rubidoux", "pedley",
            "el cerrito", "coronita", "warm springs",
            "east hemet", "valle vista", "green acres",
            "cherry valley", "sage", "garner valley",
            "lakeview", "march air reserve base", "march arb",
            "spring valley lake", "de luz", "gavilan hills",
            "meadowbrook", "quail valley", "mountain center",
            "pine cove", "pinyon pines",
        },
    },

    # --- 7 new counties ---
    "Los Angeles": {
        "bbox": (33.70, 34.82, -118.95, -117.65),
        "cities": {
            # --- Incorporated cities (88 cities) ---
            "los angeles", "long beach", "glendale", "santa clarita",
            "palmdale", "lancaster", "pomona", "torrance", "pasadena",
            "el monte", "downey", "inglewood", "west covina", "norwalk",
            "burbank", "compton", "south gate", "carson", "santa monica",
            "whittier", "hawthorne", "alhambra", "lakewood", "bellflower",
            "baldwin park", "lynwood", "redondo beach", "pico rivera",
            "montebello", "monterey park", "gardena", "huntington park",
            "arcadia", "diamond bar", "paramount", "rosemead", "glendora",
            "covina", "west hollywood", "azusa", "claremont", "la mirada",
            "rancho palos verdes", "san dimas", "monrovia", "la verne",
            "bell gardens", "duarte", "walnut", "temple city",
            "culver city", "bell", "manhattan beach", "beverly hills",
            "cerritos", "san fernando", "la canada flintridge",
            "la canada", "la cañada flintridge", "hermosa beach",
            "calabasas", "agoura hills",
            "artesia", "lomita", "south pasadena", "maywood",
            "sierra madre", "la puente", "irwindale", "lawndale",
            "bradbury", "signal hill", "palos verdes estates",
            "rolling hills estates", "rolling hills", "hidden hills",
            "industry", "city of industry", "commerce", "cudahy",
            "hawaiian gardens", "la habra heights", "san marino",
            "south el monte", "vernon", "westlake village",
            # Research-verified additions
            "avalon", "el segundo", "malibu", "san gabriel",
            "santa fe springs",
            # --- Major CDPs & unincorporated communities ---
            "altadena", "east los angeles", "east la", "florence-graham",
            "florence", "graham", "ladera heights", "lennox", "marina del rey",
            "topanga", "willowbrook", "west athens", "west carson",
            "west rancho dominguez", "rancho dominguez", "del aire",
            "el camino village", "harbor city", "harbor gateway",
            "san pedro", "wilmington", "watts", "westchester",
            "playa del rey", "playa vista", "venice", "mar vista",
            "palms", "cheviot hills", "bel air", "brentwood",
            "pacific palisades", "encino", "tarzana", "woodland hills",
            "canoga park", "chatsworth", "granada hills", "northridge",
            "porter ranch", "north hills", "panorama city", "sun valley",
            "sylmar", "pacoima", "arleta", "mission hills",
            "reseda", "van nuys", "lake balboa", "north hollywood",
            "studio city", "sherman oaks", "toluca lake", "valley village",
            "valley glen", "eagle rock", "highland park", "glassell park",
            "mount washington", "atwater village", "silver lake",
            "alondra park", "east pasadena", "east san gabriel",
            "kagel canyon", "la crescenta-montrose", "la crescenta",
            "north el monte", "south san gabriel", "universal city",
            "walnut park", "east rancho dominguez",
            "echo park", "los feliz", "koreatown", "westlake",
            "downtown los angeles", "dtla", "boyle heights",
            "lincoln heights", "el sereno", "city terrace",
            "east hollywood", "thai town", "little armenia",
            "hancock park", "larchmont", "mid-wilshire", "miracle mile",
            "carthay", "windsor square", "mid-city", "leimert park",
            "crenshaw", "hyde park", "view park", "view park-windsor hills",
            "windsor hills", "baldwin hills", "ladera",
            "south los angeles", "south la", "vermont knolls",
            "westmont", "athens", "unincorporated compton",
            "rowland heights", "hacienda heights", "avocado heights",
            "bassett", "valinda", "west puente valley", "charter oak",
            "citrus", "vincent", "west whittier", "west whittier-los nietos",
            "los nietos", "east whittier", "south whittier",
            "north whittier", "whittier narrows",
            "east rancho dominguez", "quartz hill",
            "sun village", "lake los angeles", "littlerock", "pearblossom",
            "juniper hills", "llano", "valyermo", "acton", "agua dulce",
            "castaic", "stevenson ranch", "val verde", "newhall",
            "canyon country", "saugus", "gorman", "lebec",
            "lake hughes", "elizabeth lake", "green valley", "valencia",
            "west hills", "marina del rey",
            "leona valley", "lago vista", "three points",
        },
    },
    "Orange": {
        "bbox": (33.38, 33.95, -118.11, -117.41),
        "cities": {
            # --- All 34 incorporated cities ---
            "anaheim", "santa ana", "irvine", "huntington beach",
            "garden grove", "fullerton", "orange", "costa mesa",
            "mission viejo", "westminster", "newport beach", "buena park",
            "lake forest", "tustin", "yorba linda", "san clemente",
            "laguna niguel", "la habra", "fountain valley", "fountian valley",
            "placentia",
            "rancho santa margarita", "aliso viejo", "cypress", "brea",
            "stanton", "san juan capistrano", "dana point", "laguna beach",
            "laguna hills", "seal beach", "los alamitos", "villa park",
            "la palma", "laguna woods",
            # --- Major CDPs & unincorporated communities ---
            "coto de caza", "ladera ranch", "las flores", "foothill ranch",
            "portola hills", "trabuco canyon", "dove canyon",
            "robinson ranch", "rancho mission viejo",
            "midway city", "rossmoor", "north tustin",
            "orange park acres", "silverado", "modjeska canyon",
            "trabuco highlands", "el modena", "santiago canyon",
            "emerald bay", "laguna canyon", "crystal cove",
            "capistrano beach", "monarch beach", "three arch bay",
            "south laguna", "sunset beach", "surfside",
            "balboa", "balboa island", "corona del mar", "lido isle",
            "east irvine", "woodbridge", "university park",
            "turtle rock", "shady canyon", "quail hill",
            "talega", "forster ranch", "san clemente island",
            "cowan heights", "lemon heights", "santa ana heights",
            "corona del mar",
        },
    },
    "San Bernardino": {
        # Note: largest county in contiguous US. Eastern portion is mostly
        # uninhabited desert (Mojave, Death Valley adj.). Population is
        # concentrated in the western "Inland Empire" portion.
        "bbox": (34.00, 35.81, -117.65, -114.13),
        "cities": {
            # --- Incorporated cities ---
            "san bernardino", "fontana", "rancho cucamonga", "ontario",
            "victorville", "rialto", "hesperia", "chino", "upland",
            "apple valley", "redlands", "chino hills", "highland",
            "colton", "yucaipa", "montclair", "barstow", "loma linda",
            "twentynine palms", "yucca valley", "adelanto",
            "grand terrace", "big bear lake", "needles",
            # --- Major CDPs & unincorporated communities ---
            "big bear city", "running springs", "lake arrowhead",
            "crestline", "cedar glen", "blue jay", "skyforest",
            "twin peaks", "rimforest", "green valley lake",
            "forest falls", "angelus oaks", "mentone", "loma linda",
            "devore", "devore heights", "glen helen", "lytle creek",
            "wrightwood", "phelan", "pinon hills", "oak hills",
            "spring valley lake", "oro grande", "helendale",
            "silver lakes", "lenwood", "hodge", "daggett",
            "yermo", "newberry springs", "harvard", "ludlow",
            "amboy", "cadiz", "chambless", "essex", "goffs",
            "kelso", "nipton", "primm", "baker",
            "mountain pass", "jean", "ivanpah",
            "fort irwin", "trona", "searles valley",
            "joshua tree", "morongo valley", "landers", "johnson valley",
            "flamingo heights", "pioneertown", "rimrock",
            "wonder valley", "lucerne valley", "sugarloaf",
            "fawnskin", "baldwin lake", "big river",
            "arrowbear lake", "cedarpines park", "el mirage", "hinkley",
            "mt. baldy",
            "earp", "parker dam", "havasu lake",
            "bloomington", "muscoy", "arrowhead farms",
            "del rosa", "san antonio heights", "mt baldy",
            "mount baldy", "claremont", "la verne",
            "etiwanda", "alta loma", "cucamonga",
        },
    },
    "Ventura": {
        "bbox": (34.00, 34.90, -119.47, -118.63),
        "cities": {
            # --- All 10 incorporated cities ---
            "ventura", "san buenaventura", "oxnard", "thousand oaks",
            "simi valley", "camarillo", "moorpark", "santa paula",
            "fillmore", "port hueneme", "ojai",
            # --- Major CDPs & unincorporated communities ---
            "oak park", "newbury park", "westlake village",
            "lake sherwood", "hidden valley", "casa conejo",
            "channel islands beach", "el rio", "saticoy",
            "meiners oaks", "mira monte", "oak view",
            "upper ojai", "piru", "bardsdale",
            "somis", "santa rosa valley", "bell canyon",
            "wood ranch", "brandeis", "happy camp canyon",
            "lockwood valley", "frazier park", "lake of the woods",
            "pine mountain club", "pinetree",
            "point mugu", "naval base ventura county",
            "channel islands harbor", "silver strand",
            "montalvo", "nyeland acres", "garden acres",
            "solromar", "la conchita", "santa susana",
        },
    },
    "Santa Barbara": {
        "bbox": (34.38, 35.10, -120.64, -119.22),
        "cities": {
            # --- All 8 incorporated cities ---
            "santa barbara", "santa maria", "lompoc", "goleta",
            "carpinteria", "guadalupe", "solvang", "buellton",
            # --- Major CDPs & unincorporated communities ---
            "montecito", "summerland", "isla vista", "el capitan",
            "hope ranch", "mission canyon", "san roque",
            "los alamos", "los olivos", "santa ynez", "ballard",
            "casmalia", "gaviota", "cuyama", "new cuyama",
            "orcutt", "vandenberg village", "mission hills",
            "vandenberg space force base", "vandenberg afb",
            "vandenberg", "lompoc valley",
            "nipomo", "arroyo grande",  # these are SLO but border-adjacent
            "goleta old town", "noleta",
            "eastern goleta valley", "garey", "sisquoc", "naples",
            "painted cave", "surf", "ventucopa",
        },
    },
    "Kern": {
        "bbox": (34.79, 35.79, -119.86, -117.63),
        "cities": {
            # --- All 11 incorporated cities ---
            "bakersfield", "delano", "ridgecrest", "wasco",
            "shafter", "arvin", "tehachapi", "california city",
            "taft", "maricopa", "mcfarland",
            # --- Major CDPs & unincorporated communities ---
            "lamont", "rosamond", "mojave", "boron", "edwards",
            "north edwards", "edwards afb", "edwards air force base",
            "inyokern", "johannesburg", "randsburg", "saltdale",
            "cantil", "garlock", "red rock canyon",
            "stallion springs", "bear valley springs",
            "golden hills", "keene", "monolith",
            "caliente", "bodfish", "lake isabella", "kernville",
            "wofford heights", "south lake", "weldon",
            "onyx", "walker basin", "havilah",
            "glennville", "woody", "posey",
            "lebec", "frazier park", "gorman",
            "grapevine", "wheeler ridge", "mettler",
            "greenfield", "oildale", "east bakersfield",
            "cottonwood", "lost hills", "buttonwillow",
            "tupman", "elk hills", "fellows", "mckittrick",
            "derby acres", "dustin acres", "ford city",
            "taft heights", "south taft",
            "terra bella", "ducor", "richgrove", "earlimart",
            "pixley", "allensworth", "alpaugh",
            "pond", "old river", "panama",
            "edison", "di giorgio", "arvin",
            "weedpatch", "lamont", "vineland",
            "alta sierra", "china lake acres", "mountain mesa",
            "rosedale", "squirrel mountain valley", "valley acres",
        },
    },
    "San Luis Obispo": {
        "bbox": (34.99, 35.80, -121.40, -119.47),
        "cities": {
            # --- All 7 incorporated cities ---
            "san luis obispo", "paso robles", "el paso de robles",
            "atascadero", "arroyo grande", "grover beach",
            "pismo beach", "morro bay",
            # --- Major CDPs & unincorporated communities ---
            "cambria", "los osos", "templeton", "nipomo",
            "oceano", "cayucos", "san simeon", "avila beach",
            "shell beach", "baywood-los osos", "baywood park",
            "shandon", "san miguel", "creston", "santa margarita",
            "garden farms", "edna", "edna valley",
            "pozo", "lopez lake", "huasna",
            "heritage ranch", "lake nacimiento",
            "bradley", "camp san luis obispo",
            "california valley", "carrizo plain",
            "cholame", "parkfield", "whitley gardens",
            "blacklake", "callender", "los berros", "los ranchos",
            "oak shores", "woodlands",
        },
    },
}

session = requests.Session()
session.headers.update(HEADERS)


def fetch(lat, lng, radius=RADIUS_METERS):
    url = f"{BASE_URL}/{lat}/{lng}/{radius}"
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401:
                raise RuntimeError(
                    "401 Unauthorized — subscription key rejected. "
                    "Re-check BATTERYNETWORK_APIM_KEY."
                )
            if resp.status_code == 429:
                print("  rate limited, backing off...")
                time.sleep(5 * (attempt + 1))
                continue
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException as e:
            print(f"  request error: {e}")
            time.sleep(1.5 * (attempt + 1))
    return None


def generate_grid_points(lat_min, lat_max, lng_min, lng_max, radius_m):
    """Tile query points across the bounding box with overlap."""
    lat_step_deg = (radius_m * OVERLAP_FACTOR) / 111_320
    mid_lat_rad = math.radians((lat_min + lat_max) / 2)
    lng_step_deg = (radius_m * OVERLAP_FACTOR) / (111_320 * math.cos(mid_lat_rad))

    points = []
    lat = lat_min
    while lat <= lat_max:
        lng = lng_min
        while lng <= lng_max:
            points.append((round(lat, 6), round(lng, 6)))
            lng += lng_step_deg
        lat += lat_step_deg
    return points


def parse_record(raw):
    """
    Field names confirmed via --probe (2026-07-12).
    API returns a flat list of objects with camelCase keys.
    """
    return {
        "account_number": raw.get("accountNumber", ""),
        "name": raw.get("businessName", ""),
        "street": raw.get("addressStreet", ""),
        "city": raw.get("addressCity", ""),
        "state": raw.get("addressStateProvince", ""),
        "zip": raw.get("addressZIP", ""),
        "phone": raw.get("mainPhone", ""),
        "materials": raw.get("acceptedMaterials", ""),
        "lat": raw.get("addressLatitude", ""),
        "lng": raw.get("addressLongitude", ""),
        "google_place_id": raw.get("googlePlaceID", ""),
        "hours_mon": raw.get("hoursMon", ""),
        "hours_tue": raw.get("hoursTue", ""),
        "hours_wed": raw.get("hoursWed", ""),
        "hours_thu": raw.get("hoursThu", ""),
        "hours_fri": raw.get("hoursFri", ""),
        "hours_sat": raw.get("hoursSat", ""),
        "hours_sun": raw.get("hoursSun", ""),
        "drop_off_notes": raw.get("dropOffNotes", ""),
        "raw_json": json.dumps(raw),  # keep everything for later re-parsing
    }


def in_county(rec, county_cfg):
    """Check if a record belongs to the given county by city name or bbox."""
    city = (rec.get("city") or "").strip().lower()
    if city:
        return city in county_cfg["cities"]
    # fall back to bounding-box check if city is missing
    lat_min, lat_max, lng_min, lng_max = county_cfg["bbox"]
    try:
        lat, lng = float(rec["lat"]), float(rec["lng"])
        return lat_min <= lat <= lat_max and lng_min <= lng <= lng_max
    except (TypeError, ValueError, KeyError):
        return True  # keep it if we truly can't tell; filter manually later


def probe():
    lat, lng = 33.0227476, -117.1382404  # the sample point you captured
    print(f"Probing {lat}, {lng} at {RADIUS_METERS}m radius...")
    data = fetch(lat, lng)
    print(json.dumps(data, indent=2)[:4000])
    print(
        "\nCompare these keys against parse_record()'s field name guesses "
        "and edit that function to match before running the full sweep."
    )


def full_sweep(out_path="socal_battery_locations_live.csv"):
    total_grid_points = 0
    for county_name, cfg in COUNTIES.items():
        lat_min, lat_max, lng_min, lng_max = cfg["bbox"]
        pts = generate_grid_points(lat_min, lat_max, lng_min, lng_max, RADIUS_METERS)
        total_grid_points += len(pts)

    print(f"Southern California Battery Network Sweep")
    print(f"==========================================")
    print(f"Counties: {len(COUNTIES)}")
    print(f"Total grid points: {total_grid_points}")
    print(f"Estimated time: ~{total_grid_points * (REQUEST_DELAY_SECONDS + 0.3):.0f}s")

    seen_ids = set()
    records = []
    county_stats = {}
    total_queries = 0

    for county_name, cfg in COUNTIES.items():
        lat_min, lat_max, lng_min, lng_max = cfg["bbox"]
        points = generate_grid_points(lat_min, lat_max, lng_min, lng_max, RADIUS_METERS)
        print(f"\n{'='*60}")
        print(f"{county_name} County \u2014 {len(points)} grid points")
        print(f"{'='*60}")

        county_new = 0
        for i, (lat, lng) in enumerate(points, 1):
            total_queries += 1
            print(f"  [{i}/{len(points)}] querying {lat}, {lng}")
            data = fetch(lat, lng)
            if not data:
                continue

            items = data if isinstance(data, list) else []

            for raw in items:
                rec = parse_record(raw)
                key = rec["account_number"] or (rec["street"], rec["zip"])
                if key in seen_ids:
                    continue
                if in_county(rec, cfg):
                    seen_ids.add(key)
                    rec["county"] = county_name
                    records.append(rec)
                    county_new += 1

            time.sleep(REQUEST_DELAY_SECONDS)

        county_stats[county_name] = county_new
        print(f"  \u2192 {county_new} new unique locations in {county_name} County")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total API queries: {total_queries}")
    print(f"Total unique locations: {len(records)}")
    print()
    for county, count in county_stats.items():
        print(f"  {county:.<30} {count:>4}")
    print(f"  {'TOTAL':.<30} {len(records):>4}")

    fieldnames = [
        "county", "account_number", "name", "street", "city", "state", "zip",
        "phone", "materials", "lat", "lng", "google_place_id",
        "hours_mon", "hours_tue", "hours_wed", "hours_thu",
        "hours_fri", "hours_sat", "hours_sun",
        "drop_off_notes", "raw_json",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep Southern California for Battery Network drop-off locations"
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Make one test call and print raw JSON to inspect the schema",
    )
    parser.add_argument(
        "--counties", type=str, default=None,
        help="Comma-separated list of county names to sweep (default: all)",
    )
    args = parser.parse_args()

    if args.probe:
        probe()
    elif args.counties:
        # Filter to requested counties only
        requested = {c.strip().title() for c in args.counties.split(",")}
        filtered = {k: v for k, v in COUNTIES.items() if k in requested}
        missing = requested - set(filtered.keys())
        if missing:
            print(f"Warning: unknown counties: {missing}")
            print(f"Available: {', '.join(COUNTIES.keys())}")
        if filtered:
            orig = COUNTIES.copy()
            COUNTIES.clear()
            COUNTIES.update(filtered)
            full_sweep()
            COUNTIES.clear()
            COUNTIES.update(orig)
    else:
        full_sweep()
