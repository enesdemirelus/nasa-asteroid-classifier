import { NextResponse } from "next/server";

const NASA_API_KEY = process.env.NASA_API_KEY ?? "DEMO_KEY";
const NASA_FEED_URL = "https://api.nasa.gov/neo/rest/v1/feed";
const NASA_NEO_URL = "https://api.nasa.gov/neo/rest/v1/neo";

function formatDate(date: Date): string {
  return date.toISOString().split("T")[0];
}

interface NasaFeedObject {
  id: string;
  name: string;
  close_approach_data: {
    epoch_date_close_approach: number;
  }[];
}

interface NasaFeedResponse {
  near_earth_objects: Record<string, NasaFeedObject[]>;
}

interface NasaDetailResponse {
  name: string;
  absolute_magnitude_h: number;
  estimated_diameter: {
    kilometers: {
      estimated_diameter_min: number;
      estimated_diameter_max: number;
    };
  };
  is_potentially_hazardous_asteroid: boolean;
  close_approach_data: {
    relative_velocity: { kilometers_per_second: string };
    miss_distance: { kilometers: string };
  }[];
  orbital_data: {
    orbit_uncertainty: string;
    minimum_orbit_intersection: string;
    jupiter_tisserand_invariant: string;
    eccentricity: string;
    semi_major_axis: string;
    inclination: string;
    ascending_node_longitude: string;
    orbital_period: string;
    perihelion_distance: string;
    perihelion_argument: string;
    aphelion_distance: string;
    perihelion_time: string;
    mean_anomaly: string;
    mean_motion: string;
  };
}

async function getPrediction(asteroidData: Record<string, unknown>) {
  const res = await fetch(
    "https://nasa-asteroid-classifier-production.up.railway.app/predict",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: asteroidData }),
    },
  );
  if (!res.ok) throw new Error(`Prediction API error: ${res.status}`);
  const { hazard_probability, predicted_hazardous } = await res.json();
  return { hazard_probability, predicted_hazardous };
}

async function getAsteroidIdsByDate(
  startDate: string,
  endDate: string,
): Promise<{ id: string; epochDate: number }[]> {
  const url = new URL(NASA_FEED_URL);
  url.searchParams.set("start_date", startDate);
  url.searchParams.set("end_date", endDate);
  url.searchParams.set("api_key", NASA_API_KEY);

  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`NASA feed API error: ${res.status}`);

  const data: NasaFeedResponse = await res.json();
  const entries: { id: string; epochDate: number }[] = [];

  for (const date of Object.keys(data.near_earth_objects)) {
    for (const obj of data.near_earth_objects[date]) {
      const epochDate =
        obj.close_approach_data[0]?.epoch_date_close_approach ?? 0;
      entries.push({ id: obj.id, epochDate });
    }
  }

  return entries;
}

async function getCompleteAsteroidData(asteroidId: string) {
  const url = new URL(`${NASA_NEO_URL}/${asteroidId}`);
  url.searchParams.set("api_key", NASA_API_KEY);

  const res = await fetch(url.toString());
  if (!res.ok)
    throw new Error(`NASA NEO API error: ${res.status} for id ${asteroidId}`);

  const d: NasaDetailResponse = await res.json();

  if (!d.close_approach_data?.length)
    throw new Error(`No close approach data for asteroid ${asteroidId}`);
  if (!d.orbital_data)
    throw new Error(`No orbital data for asteroid ${asteroidId}`);

  const approach = d.close_approach_data[0];
  const orbit = d.orbital_data;

  return {
    Name: d.name,
    "Absolute Magnitude": d.absolute_magnitude_h,
    "Est Dia in KM(min)":
      d.estimated_diameter.kilometers.estimated_diameter_min,
    "Est Dia in KM(max)":
      d.estimated_diameter.kilometers.estimated_diameter_max,
    "Relative Velocity km per sec": parseFloat(
      approach.relative_velocity.kilometers_per_second,
    ),
    "Miss Dist.(kilometers)": parseFloat(approach.miss_distance.kilometers),
    "Orbit Uncertainity": parseInt(orbit.orbit_uncertainty),
    "Minimum Orbit Intersection": parseFloat(orbit.minimum_orbit_intersection),
    "Jupiter Tisserand Invariant": parseFloat(
      orbit.jupiter_tisserand_invariant,
    ),
    Eccentricity: parseFloat(orbit.eccentricity),
    "Semi Major Axis": parseFloat(orbit.semi_major_axis),
    Inclination: parseFloat(orbit.inclination),
    "Asc Node Longitude": parseFloat(orbit.ascending_node_longitude),
    "Orbital Period": parseFloat(orbit.orbital_period),
    "Perihelion Distance": parseFloat(orbit.perihelion_distance),
    "Perihelion Arg": parseFloat(orbit.perihelion_argument),
    "Aphelion Dist": parseFloat(orbit.aphelion_distance),
    "Perihelion Time": parseFloat(orbit.perihelion_time),
    "Mean Anomaly": parseFloat(orbit.mean_anomaly),
    "Mean Motion": parseFloat(orbit.mean_motion),
    Hazardous: d.is_potentially_hazardous_asteroid,
  };
}

export async function GET() {
  try {
    const endDate = new Date();
    const startDate = new Date(endDate);
    startDate.setDate(endDate.getDate() - 7);

    const entries = await getAsteroidIdsByDate(
      formatDate(startDate),
      formatDate(endDate),
    );

    const recent5 = entries
      .sort((a, b) => b.epochDate - a.epochDate)
      .slice(0, 15);

    const results = await Promise.allSettled(
      recent5.map(async ({ id }) => {
        const data = await getCompleteAsteroidData(id);
        const prediction = await getPrediction(data);
        return {
          ...data,
          "Predicted Hazardous": prediction.predicted_hazardous,
          "Hazard Probability": prediction.hazard_probability,
        };
      }),
    );

    const asteroids = results
      .filter(
        (
          r,
        ): r is PromiseFulfilledResult<
          typeof r extends PromiseFulfilledResult<infer T> ? T : never
        > => {
          if (r.status === "rejected") {
            console.error("[get-asteroids] asteroid fetch failed:", r.reason);
          }
          return r.status === "fulfilled";
        },
      )
      .map((r) => r.value);

    return NextResponse.json({ asteroids });
  } catch (error) {
    console.error("[get-asteroids] route error:", error);
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
