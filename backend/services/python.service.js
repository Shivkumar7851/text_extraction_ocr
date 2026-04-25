import { spawn }  from "child_process";
import path        from "path";
import fs          from "fs";
import os          from "os";

const PIPELINE_PATH = path.resolve("../python_pipeline/ocr_pipeline.py");
const OUTPUT_DIR    = path.resolve("../python_pipeline/ocr_outputs");

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper — spawn python3 and return parsed JSON from stdout
// ─────────────────────────────────────────────────────────────────────────────

function _spawnPipeline(args) {
  return new Promise((resolve, reject) => {
    const python = spawn("python3", [PIPELINE_PATH, ...args]);

    let stdout = "";

    python.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    python.stderr.on("data", (chunk) => { process.stderr.write(`[ocr-pipeline] ${chunk}`); });

    python.on("error", (err) => {
      reject(new Error(`Failed to start python3: ${err.message}`));
    });

    python.on("close", (code) => {
      let parsed;
      try {
        parsed = JSON.parse(stdout);
      } catch {
        return reject(new Error(
          `Pipeline returned non-JSON output (exit ${code}):\n${stdout.slice(0, 500)}`
        ));
      }

      if (code !== 0) {
        return reject(Object.assign(
          new Error(parsed.error ?? `Pipeline exited with code ${code}`),
          { pipelineResponse: parsed }
        ));
      }

      resolve(parsed);
    });
  });
}

/**
 * @param {string|string[]} filePaths
 * @param {object}  [opts]
 * @param {string}  [opts.type]         'printed' | 'handwritten' | 'mixed'
 * @param {string}  [opts.outputDir]
 * @param {string}  [opts.groundTruth]  Reference text → enables absolute_accuracy in result.
 *                                      Only valid for a single file.
 */
export const runPythonPipeline = (filePaths, opts = {}) => {
  const files  = Array.isArray(filePaths) ? filePaths : [filePaths];
  const outDir = opts.outputDir ?? OUTPUT_DIR;

  const gt = typeof opts.groundTruth === "string" && opts.groundTruth.trim()
    ? opts.groundTruth
    : null;

  if (gt && files.length > 1) {
    return Promise.reject(
      new Error("runPythonPipeline: opts.groundTruth can only be used with a single file")
    );
  }

  const args = [
    ...files,
    "--output-dir", outDir,
    ...(opts.type ? ["--type", opts.type] : []),
    ...(gt        ? ["--ground-truth", gt] : []),
  ];

  return _spawnPipeline(args);
};

// ─────────────────────────────────────────────────────────────────────────────
// runWithGroundTruth  — convenience wrapper
// ─────────────────────────────────────────────────────────────────────────────

export const runWithGroundTruth = (filePath, groundTruth, opts = {}) => {
  if (typeof filePath    !== "string" || !filePath.trim())
    return Promise.reject(new Error("runWithGroundTruth: filePath must be a non-empty string"));
  if (typeof groundTruth !== "string")
    return Promise.reject(new Error("runWithGroundTruth: groundTruth must be a string"));

  return runPythonPipeline(filePath, { ...opts, groundTruth });
};

// ─────────────────────────────────────────────────────────────────────────────
// runBatchWithGroundTruth
// ─────────────────────────────────────────────────────────────────────────────

export const runBatchWithGroundTruth = async (records, opts = {}) => {
  if (!Array.isArray(records) || records.length === 0)
    throw new Error("runBatchWithGroundTruth: records must be a non-empty array");

  for (const rec of records) {
    if (typeof rec.image !== "string" || !rec.image.trim())
      throw new Error(`runBatchWithGroundTruth: every record needs a non-empty "image" field. Got: ${JSON.stringify(rec)}`);
    if (rec.ground_truth !== undefined && typeof rec.ground_truth !== "string")
      throw new Error(`runBatchWithGroundTruth: "ground_truth" must be a string. Got: ${JSON.stringify(rec)}`);
  }

  const outDir  = opts.outputDir ?? OUTPUT_DIR;
  const tmpFile = path.join(os.tmpdir(), `ocr_gt_${Date.now()}_${process.pid}.json`);

  try {
    fs.writeFileSync(tmpFile, JSON.stringify(records), "utf8");
    const args = [
      "--gt-file",    tmpFile,
      "--output-dir", outDir,
      ...(opts.type ? ["--type", opts.type] : []),
    ];
    return await _spawnPipeline(args);
  } finally {
    try { fs.unlinkSync(tmpFile); } catch { /* ignore */ }
  }
};