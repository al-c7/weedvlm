/**
 * The benchmark composition config: for each question type, how many questions to keep per
 * benchmark_class ("questions_per_class") and how many classes to keep ("num_classes"). Both the
 * GUI and the tracked benchmark-config.json file edit exactly this shape, so either one can be the
 * source of truth at any given time -- "Save config" just persists whatever the GUI currently has.
 */

export interface TaskConfig {
  questions_per_class: number;
  num_classes: number;
}

export interface BenchmarkConfig {
  seed: number;
  exclude_flagged: boolean;
  tasks: Record<string, TaskConfig>;
}

// Ships as the tracked apps/benchmark-builder/benchmark-config.json. Sized so that, added up
// across the tasks that only differ by species (species_id_multiple_choice, species_id_open_ended,
// species_localisation), the overall benchmark stays in the same few-hundred-questions ballpark as
// fine_grained_id and density_estimation, whose classes are a fixed small vocabulary rather than
// however many species happen to be reviewed.
export const DEFAULT_CONFIG: BenchmarkConfig = {
  seed: 42,
  exclude_flagged: true,
  tasks: {
    species_id_multiple_choice: { questions_per_class: 18, num_classes: 25 },
    species_id_open_ended: { questions_per_class: 5, num_classes: 15 },
    fine_grained_id: { questions_per_class: 150, num_classes: 2 },
    species_localisation: { questions_per_class: 18, num_classes: 25 },
    density_estimation: { questions_per_class: 40, num_classes: 4 },
  },
};

export async function loadConfig(path: string): Promise<BenchmarkConfig> {
  try {
    const text = await Deno.readTextFile(path);
    return JSON.parse(text) as BenchmarkConfig;
  } catch {
    return DEFAULT_CONFIG;
  }
}

export async function saveConfig(path: string, config: BenchmarkConfig): Promise<void> {
  const tmp = `${path}.tmp`;
  await Deno.writeTextFile(tmp, JSON.stringify(config, null, 2) + "\n");
  await Deno.rename(tmp, path);
}
