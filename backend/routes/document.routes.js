// routes/upload.js  (Express)
// Handles POST /api/upload
// Accepts:  multipart/form-data  with fields:
//   file         (required) — the image file
//   groundTruth  (optional) — reference text for absolute accuracy measurement

import express    from "express";
import multer     from "multer";
import path       from "path";
import { runPythonPipeline } from "../services/python.service.js";

const router  = express.Router();


const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename:    (req, file, cb) => cb(null, `${Date.now()}_${file.originalname}`),
});
const upload = multer({ storage });

router.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: "No file uploaded" });
    }

    const filePath    = path.resolve(req.file.path);

    const groundTruth = typeof req.body.groundTruth === "string" && req.body.groundTruth.trim()
      ? req.body.groundTruth.trim()
      : undefined;


    const result = await runPythonPipeline(filePath, {
      groundTruth,
    });

    return res.json({ success: true, data: result });

  } catch (err) {
    console.error("[upload] pipeline error:", err);
    return res.status(500).json({
      success: false,
      error: err.message,
      ...(err.pipelineResponse ? { pipeline: err.pipelineResponse } : {}),
    });
  }
});

export default router;