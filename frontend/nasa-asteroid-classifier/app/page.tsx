"use client";
import {
  Badge,
  Button,
  Card,
  Divider,
  Group,
  Loader,
  Paper,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { useState } from "react";
import axios from "axios";

interface Asteroid {
  Name: string;
  Hazardous: boolean;
  "Predicted Hazardous": boolean;
  "Hazard Probability": number;
  "Absolute Magnitude": number;
  "Est Dia in KM(min)": number;
  "Est Dia in KM(max)": number;
  "Relative Velocity km per sec": number;
  "Miss Dist.(kilometers)": number;
  "Orbital Period": number;
  Eccentricity: number;
  Inclination: number;
}

function formatDistance(km: number) {
  if (km >= 1_000_000) return `${(km / 1_000_000).toFixed(2)}M km`;
  if (km >= 1_000) return `${(km / 1_000).toFixed(1)}K km`;
  return `${km.toFixed(0)} km`;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Stack gap={2}>
      <Text size="xs" c="dimmed" tt="uppercase" lts="0.05em">
        {label}
      </Text>
      <Text size="sm" fw={600} ff="monospace">
        {value}
      </Text>
    </Stack>
  );
}

function AsteroidCard({ asteroid }: { asteroid: Asteroid }) {
  const isHazardous = asteroid["Predicted Hazardous"];
  const confidence = isHazardous
    ? asteroid["Hazard Probability"] * 100
    : (1 - asteroid["Hazard Probability"]) * 100;
  const avgDia =
    (asteroid["Est Dia in KM(min)"] + asteroid["Est Dia in KM(max)"]) / 2;

  return (
    <Card
      padding="lg"
      radius="md"
      style={{
        backgroundColor: "#0d1129",
        border: "1px solid #1a2044",
      }}
    >
      <Stack gap="md">
        <Group justify="space-between" align="flex-start" wrap="nowrap">
          <Text fw={600} size="md">
            {asteroid.Name}
          </Text>
          <Badge
            color={isHazardous ? "red" : "green"}
            variant="light"
            size="lg"
            style={{ flexShrink: 0 }}
          >
            {isHazardous ? "Hazardous" : "Safe"}
          </Badge>
        </Group>

        <Text size="sm" c="dimmed" ff="monospace">
          {confidence.toFixed(1)}% confidence
        </Text>

        <Divider color="#1a2044" />

        <SimpleGrid cols={2} spacing="sm" verticalSpacing="sm">
          <Stat
            label="Miss Distance"
            value={formatDistance(asteroid["Miss Dist.(kilometers)"])}
          />
          <Stat
            label="Velocity"
            value={`${asteroid["Relative Velocity km per sec"].toFixed(2)} km/s`}
          />
          <Stat label="Avg Diameter" value={`${avgDia.toFixed(3)} km`} />
          <Stat
            label="Orbital Period"
            value={`${asteroid["Orbital Period"].toFixed(0)} days`}
          />
        </SimpleGrid>

        <Divider color="#1a2044" />

        <Group gap="xs">
          <Text size="xs" c="dimmed">
            NASA:
          </Text>
          <Badge
            color={asteroid.Hazardous ? "orange" : "gray"}
            variant="light"
            size="sm"
          >
            {asteroid.Hazardous ? "Potentially Hazardous" : "Not Hazardous"}
          </Badge>
        </Group>
      </Stack>
    </Card>
  );
}

const DEMO_ASTEROID: Asteroid = {
  Name: "Demo Object — 1997 XF11",
  Hazardous: true,
  "Predicted Hazardous": true,
  "Hazard Probability": 0.9863207936286926,
  "Absolute Magnitude": 21.6,
  "Est Dia in KM(min)": 0.12722,
  "Est Dia in KM(max)": 0.284472,
  "Relative Velocity km per sec": 6.115834,
  "Miss Dist.(kilometers)": 62753692.0,
  "Orbital Period": 609.599786,
  Eccentricity: 0.425549,
  Inclination: 6.025981,
};

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

  const loadDemo = () => {
    setAsteroids([DEMO_ASTEROID]);
    setError(null);
  };

  const safeCount = asteroids.filter((a) => !a["Predicted Hazardous"]).length;
  const hazardousCount = asteroids.filter(
    (a) => a["Predicted Hazardous"],
  ).length;

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#06091a",
        padding: "48px 24px",
        position: "relative",
      }}
    >
      <div className="starfield" />

      <Stack
        gap="xl"
        maw={1100}
        mx="auto"
        style={{ position: "relative", zIndex: 1 }}
      >
        {/* Header */}
        <Stack gap="xs">
          <Text size="xs" ff="monospace" c="dimmed" tt="uppercase" lts="0.15em">
            NASA Near-Earth Object Classification
          </Text>
          <Title order={2}>Asteroid Threat Detector</Title>
          <Stack gap={4} mt={4}>
            <Text size="sm" c="dimmed">
              Live near-Earth object data comes from the{" "}
              <Text component="span" size="sm" c="indigo.3">
                NASA NeoWs API
              </Text>
              , which covers asteroids tracked within a 7-day window. Each
              object gets run through a Keras neural network I trained on{" "}
              <Text component="span" size="sm" c="indigo.3">
                NASA&apos;s historical NEO dataset
              </Text>{" "}
              to classify it as hazardous or not based on its orbital and
              physical data.
            </Text>
            <Text size="sm" c="dimmed">
              Each card also shows the{" "}
              <Text component="span" size="sm" c="orange.4">
                NASA Status
              </Text>{" "}
              so you can see how the model compares to NASA&apos;s own
              classification.
            </Text>
          </Stack>
          <Paper
            mt="sm"
            p="sm"
            radius="sm"
            style={{ backgroundColor: "#110e1e", border: "1px solid #2d1f4e" }}
          >
            <Text size="xs" c="dimmed">
              <Text component="span" size="xs" c="yellow.4" fw={600}>
                Heads up:{" "}
              </Text>
              Most asteroids from the live feed will come back will be safe as a
              nature. If you want to see the model flag something as hazardous,
              hit the{" "}
              <Text component="span" size="xs" c="indigo.3">
                Hazardous Asteroid Demo
              </Text>{" "}
              button.
            </Text>
          </Paper>
        </Stack>

        {/* Buttons */}
        <Group>
          <Button
            onClick={getAsteroids}
            loading={loading}
            variant="filled"
            color="indigo"
          >
            Fetch Recent Asteroids
          </Button>
          <Button onClick={loadDemo} variant="light" color="red">
            Hazardous Asteroid Demo
          </Button>
        </Group>

        {error && (
          <Text size="sm" c="red">
            {error}
          </Text>
        )}

        {/* Summary row */}
        {asteroids.length > 0 && (
          <>
            <Group gap="sm">
              <Paper
                p="sm"
                radius="md"
                style={{
                  backgroundColor: "#0d1129",
                  border: "1px solid #1a2044",
                }}
              >
                <Text size="xs" c="dimmed" tt="uppercase" lts="0.05em">
                  Scanned
                </Text>
                <Text size="lg" fw={700} ff="monospace">
                  {asteroids.length}
                </Text>
              </Paper>
              <Paper
                p="sm"
                radius="md"
                style={{
                  backgroundColor: "#0d1129",
                  border: "1px solid #1a2044",
                }}
              >
                <Text size="xs" c="dimmed" tt="uppercase" lts="0.05em">
                  Safe
                </Text>
                <Text size="lg" fw={700} ff="monospace" c="green">
                  {safeCount}
                </Text>
              </Paper>
              <Paper
                p="sm"
                radius="md"
                style={{
                  backgroundColor: "#0d1129",
                  border: "1px solid #1a2044",
                }}
              >
                <Text size="xs" c="dimmed" tt="uppercase" lts="0.05em">
                  Hazardous
                </Text>
                <Text size="lg" fw={700} ff="monospace" c="red">
                  {hazardousCount}
                </Text>
              </Paper>
            </Group>

            {/* Card grid */}
            <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="md">
              {asteroids.map((asteroid) => (
                <AsteroidCard key={asteroid.Name} asteroid={asteroid} />
              ))}
            </SimpleGrid>
          </>
        )}

        {/* Empty state */}
        {!loading && asteroids.length === 0 && (
          <Text size="sm" c="dimmed">
            Click the button to scan recent asteroids.
          </Text>
        )}
      </Stack>
    </div>
  );
}
