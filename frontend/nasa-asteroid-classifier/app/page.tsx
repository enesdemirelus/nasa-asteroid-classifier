"use client";
import { Button } from "@mantine/core";
import { useState } from "react";
import axios from "axios";

interface Asteroid {
  Name: string;
  Hazardous: boolean;
  "Predicted Hazardous": boolean;
  "Hazard Probability": number;
}

export default function Home() {
  const [asteroids, setAsteroids] = useState<Asteroid[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getAsteroids = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get("/get-asteroids");
      setAsteroids(response.data.asteroids);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Real Time NASA Asteroid Classifier</h1>
      <Button onClick={getAsteroids} loading={loading}>
        Get Data
      </Button>
      {error && <div>{error}</div>}
      <div>
        {asteroids.map((asteroid) => (
          <div key={asteroid.Name}>
            <strong>{asteroid.Name}</strong>
            <span>
              Predicted:{" "}
              {asteroid["Predicted Hazardous"] ? "Hazardous" : "Safe"} (
              {asteroid["Predicted Hazardous"]
                ? (asteroid["Hazard Probability"] * 100).toFixed(1)
                : ((1 - asteroid["Hazard Probability"]) * 100).toFixed(1)}
              % confidence)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
