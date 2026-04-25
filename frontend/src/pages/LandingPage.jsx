import { useEffect, useRef, useState } from "react";
import axios from "axios";
import toast from "react-hot-toast";

export default function LandingPage() {
    const [theme, setTheme] = useState("dark");
    const isDark = theme === "dark";

    useEffect(() => {
        const els = document.querySelectorAll(".anim-up");
        els.forEach((el, i) => {
            el.style.opacity = "0";
            el.style.transform = "translateY(20px)";
            setTimeout(() => {
                el.style.transition = "opacity 0.6s ease, transform 0.6s ease";
                el.style.opacity = "1";
                el.style.transform = "translateY(0)";
            }, i * 150);
        });
    }, []);

    const [image, setImage]             = useState(null);
    const [file, setFile]               = useState(null);
    const [result, setResult]           = useState(null);
    const [loading, setLoading]         = useState(false);
    const [copied, setCopied]           = useState(false);
    const [activeTab, setActiveTab]     = useState("text");
    const [groundTruth, setGroundTruth] = useState("");
    const fileInputRef = useRef();

    const backendUrl = "http://localhost:3000";

    const handleClick = () => fileInputRef.current.click();

    const handleFileChange = (e) => {
        const temp = e.target.files[0];
        if (temp) {
            setFile(temp);
            setImage(URL.createObjectURL(temp));
            setResult(null);
        }
    };

    const handleRemove = () => {
        setImage(null);
        setFile(null);
        setResult(null);
        setGroundTruth("");
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    const handleFindText = async () => {
        try {
            setLoading(true);
            const formData = new FormData();
            formData.append("file", file);
            if (groundTruth.trim()) formData.append("groundTruth", groundTruth.trim());

            const response = await axios.post(
                `${backendUrl}/api/upload`,
                formData,
                { headers: { "Content-Type": "multipart/form-data" } }
            );

            const data = response.data?.data;
            if (!data || !data.text?.cleaned) { toast.error("No text detected"); return; }

            toast.success("Text detected");
            setResult(data);
            setActiveTab("text");
        } catch (error) {
            console.error(error);
            toast.error(error.message);
        } finally {
            setLoading(false);
        }
    };

    const handleCopy = async () => {
        if (!result?.text?.cleaned) return;
        try {
            await navigator.clipboard.writeText(result.text.cleaned);
            setCopied(true);
            toast.success("Copied!");
            setTimeout(() => setCopied(false), 2000);
        } catch { toast.error("Failed to copy"); }
    };

    const cleanedText   = result?.text?.cleaned ?? "";
    const docType       = result?.document_type ?? "";
    const confidence    = result?.ocr?.avg_confidence ?? 0;
    const wordCount     = result?.text?.statistics?.total_words ?? result?.ocr?.word_count ?? 0;
    const lineCount     = result?.text?.line_count ?? 0;
    const sentiment     = result?.semantic_analysis?.sentiment ?? "";
    const category      = result?.semantic_analysis?.document_category ?? "";
    const keywords      = result?.semantic_analysis?.top_keywords ?? [];
    const namedEntities = result?.extracted_entities?.named_entities_spacy ?? {};
    const regexEntities = result?.extracted_entities ?? {};
    const noiseIssues   = result?.text?.noise_issues ?? [];
    const engineUsed    = result?.ocr?.ocr_engine ?? "";
    const isLowConf     = result?.ocr?.is_low_confidence ?? false;
    const confidenceScore   = result?.confidence_estimate?.estimated_accuracy_score ?? 0;
    const confidenceQuality = result?.confidence_estimate?.quality ?? "";
    const absAccuracy       = result?.absolute_accuracy ?? null;
    const absOverall        = absAccuracy?.overall_accuracy ?? null;
    const absQuality        = absAccuracy?.quality ?? "";

    const sentimentColor = { positive: "#059669", negative: "#DC2626", neutral: isDark ? "#6B7280" : "#9CA3AF" }[sentiment] ?? (isDark ? "#6B7280" : "#9CA3AF");

    const displayAccuracyScore   = absOverall !== null ? absOverall   : confidenceScore;
    const displayAccuracyQuality = absOverall !== null ? absQuality   : confidenceQuality;
    const displayAccuracyLabel   = absOverall !== null ? "Accuracy"   : "Est. Accuracy";
    const accuracyColor = { high: "#059669", medium: "#D97706", low: "#DC2626" }[displayAccuracyQuality] ?? (isDark ? "#6B7280" : "#9CA3AF");

    const tabs = ["text", "entities", "analysis"];

    // ── Theme tokens ──────────────────────────────────────────────────────────
    const t = isDark ? {
        bg:             "#0A0A0B",
        surface:        "#111113",
        surfaceHover:   "#18181C",
        card:           "rgba(0,0,0,0.55)",
        cardBorder:     "rgba(255,255,255,0.07)",
        navBorder:      "rgba(255,255,255,0.07)",
        text:           "#F4F4F5",
        textMuted:      "rgba(255,255,255,0.45)",
        textDim:        "rgba(255,255,255,0.25)",
        textFaint:      "rgba(255,255,255,0.15)",
        brand:          "#10B981",
        brandHover:     "#059669",
        brandBg:        "rgba(16,185,129,0.12)",
        brandBorder:    "rgba(16,185,129,0.28)",
        pillBg:         "rgba(255,255,255,0.06)",
        pillText:       "rgba(255,255,255,0.65)",
        inputBg:        "rgba(255,255,255,0.05)",
        inputBorder:    "rgba(255,255,255,0.10)",
        divider:        "rgba(255,255,255,0.07)",
        statBg:         "rgba(255,255,255,0.03)",
        footerText:     "#4B5563",
        gridLine:       "#10B981",
        glow:           "rgba(16,185,129,0.08)",
        toggleBg:       "rgba(255,255,255,0.08)",
        toggleIcon:     "☀️",
        toggleLabel:    "Light mode",
        warnColor:      "#DC2626",
        okColor:        "#10B981",
    } : {
        bg:             "#F8FAFC",
        surface:        "#FFFFFF",
        surfaceHover:   "#F1F5F9",
        card:           "rgba(255,255,255,0.95)",
        cardBorder:     "rgba(0,0,0,0.08)",
        navBorder:      "rgba(0,0,0,0.08)",
        text:           "#0F172A",
        textMuted:      "rgba(15,23,42,0.55)",
        textDim:        "rgba(15,23,42,0.38)",
        textFaint:      "rgba(15,23,42,0.22)",
        brand:          "#059669",
        brandHover:     "#047857",
        brandBg:        "rgba(5,150,105,0.08)",
        brandBorder:    "rgba(5,150,105,0.22)",
        pillBg:         "rgba(0,0,0,0.05)",
        pillText:       "rgba(15,23,42,0.70)",
        inputBg:        "rgba(0,0,0,0.03)",
        inputBorder:    "rgba(0,0,0,0.10)",
        divider:        "rgba(0,0,0,0.07)",
        statBg:         "rgba(0,0,0,0.02)",
        footerText:     "#94A3B8",
        gridLine:       "#059669",
        glow:           "rgba(5,150,105,0.06)",
        toggleBg:       "rgba(0,0,0,0.07)",
        toggleIcon:     "🌙",
        toggleLabel:    "Dark mode",
        warnColor:      "#DC2626",
        okColor:        "#059669",
    };

    const styles = {
        root: {
            minHeight: "100vh",
            background: t.bg,
            display: "flex",
            flexDirection: "column",
            fontFamily: "'DM Sans', system-ui, sans-serif",
            transition: "background 0.35s ease, color 0.35s ease",
            color: t.text,
        },
        nav: {
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 2.5rem",
            height: "60px",
            borderBottom: `1px solid ${t.navBorder}`,
            background: isDark ? "rgba(10,10,11,0.85)" : "rgba(255,255,255,0.90)",
            backdropFilter: "blur(12px)",
            position: "sticky",
            top: 0,
            zIndex: 50,
        },
        brandDot: {
            width: "8px", height: "8px",
            borderRadius: "50%",
            background: t.brand,
            animation: "pulse 2s infinite",
        },
        brandText: {
            fontWeight: 700,
            fontSize: "1.05rem",
            letterSpacing: "-0.02em",
            color: t.text,
            marginLeft: "0.5rem",
        },
        toggleBtn: {
            display: "flex",
            alignItems: "center",
            gap: "0.4rem",
            padding: "0.4rem 0.85rem",
            borderRadius: "999px",
            border: `1px solid ${t.cardBorder}`,
            background: t.toggleBg,
            cursor: "pointer",
            fontSize: "0.78rem",
            fontWeight: 600,
            color: t.textMuted,
            transition: "all 0.2s ease",
            userSelect: "none",
        },
        main: {
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: "3rem 1.5rem",
            position: "relative",
            overflow: "hidden",
        },
        gridOverlay: {
            position: "absolute",
            inset: 0,
            opacity: isDark ? 0.025 : 0.04,
            backgroundImage: `linear-gradient(${t.gridLine} 1px, transparent 1px), linear-gradient(90deg, ${t.gridLine} 1px, transparent 1px)`,
            backgroundSize: "48px 48px",
            pointerEvents: "none",
        },
        glow: {
            position: "absolute",
            top: "50%", left: "50%",
            transform: "translate(-50%,-50%)",
            width: "520px", height: "320px",
            background: isDark
                ? "radial-gradient(ellipse, rgba(16,185,129,0.10) 0%, transparent 70%)"
                : "radial-gradient(ellipse, rgba(5,150,105,0.08) 0%, transparent 70%)",
            pointerEvents: "none",
        },
        content: { position: "relative", maxWidth: "620px", width: "100%", textAlign: "center" },
        h1: {
            fontWeight: 800,
            fontSize: "clamp(2.4rem, 5vw, 3.5rem)",
            lineHeight: 1.07,
            letterSpacing: "-0.03em",
            color: t.text,
            marginBottom: "0.3rem",
        },
        h1Accent: { color: t.brand },
        uploadBtn: {
            background: t.brand,
            color: "#fff",
            border: "none",
            padding: "0.85rem 2rem",
            borderRadius: "14px",
            fontWeight: 700,
            fontSize: "0.9rem",
            cursor: "pointer",
            transition: "all 0.2s ease",
            letterSpacing: "-0.01em",
            boxShadow: isDark
                ? `0 0 24px rgba(16,185,129,0.25), 0 2px 8px rgba(0,0,0,0.4)`
                : `0 2px 12px rgba(5,150,105,0.25)`,
        },
        imgPreview: {
            height: "240px",
            objectFit: "cover",
            borderRadius: "14px",
            marginTop: "0.75rem",
            boxShadow: isDark
                ? "0 8px 32px rgba(0,0,0,0.5)"
                : "0 4px 20px rgba(0,0,0,0.12)",
            border: `1px solid ${t.cardBorder}`,
        },
        gtLabel: {
            display: "block",
            fontSize: "0.65rem",
            color: t.textDim,
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            marginBottom: "0.5rem",
            textAlign: "left",
        },
        gtTextarea: {
            width: "100%",
            borderRadius: "14px",
            padding: "0.85rem 1rem",
            fontSize: "0.85rem",
            color: t.text,
            resize: "none",
            outline: "none",
            background: t.inputBg,
            border: `1px solid ${t.inputBorder}`,
            transition: "border 0.2s, background 0.2s",
            fontFamily: "inherit",
            boxSizing: "border-box",
        },
        actionBtn: {
            background: t.brand,
            color: "#fff",
            border: "none",
            padding: "0.85rem 2rem",
            borderRadius: "14px",
            fontWeight: 700,
            fontSize: "0.9rem",
            cursor: "pointer",
            transition: "all 0.2s ease",
            letterSpacing: "-0.01em",
            boxShadow: isDark
                ? `0 0 20px rgba(16,185,129,0.22)`
                : `0 2px 10px rgba(5,150,105,0.22)`,
        },
        removeBtn: {
            background: isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.06)",
            color: t.textMuted,
            border: `1px solid ${t.cardBorder}`,
            padding: "0.85rem 1.5rem",
            borderRadius: "14px",
            fontWeight: 600,
            fontSize: "0.9rem",
            cursor: "pointer",
            transition: "all 0.2s ease",
        },
        card: {
            marginTop: "1.75rem",
            width: "100%",
            borderRadius: "20px",
            overflow: "hidden",
            background: t.card,
            border: `1px solid ${t.cardBorder}`,
            boxShadow: isDark
                ? "0 8px 40px rgba(0,0,0,0.45)"
                : "0 4px 24px rgba(0,0,0,0.08)",
            textAlign: "left",
            backdropFilter: "blur(16px)",
        },
        statBar: {
            display: "grid",
            gridTemplateColumns: "repeat(5, 1fr)",
            borderBottom: `1px solid ${t.divider}`,
        },
        statCell: {
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            padding: "0.85rem 0.5rem",
            borderRight: `1px solid ${t.divider}`,
        },
        statLabel: {
            fontSize: "0.6rem",
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            color: t.textDim,
            marginBottom: "2px",
        },
        statSub: { fontSize: "0.6rem", textTransform: "uppercase", color: t.textDim, marginTop: "2px" },
        statHint: { fontSize: "0.55rem", color: t.textFaint, marginTop: "1px" },
        absRow: {
            display: "grid",
            gridTemplateColumns: "repeat(4,1fr)",
            borderBottom: `1px solid ${t.divider}`,
            background: isDark ? "rgba(255,255,255,0.015)" : "rgba(0,0,0,0.015)",
        },
        absCell: {
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            padding: "0.6rem 0.5rem",
            borderRight: `1px solid ${t.divider}`,
        },
        absLabel: { fontSize: "0.55rem", textTransform: "uppercase", letterSpacing: "0.08em", color: t.textFaint, marginBottom: "2px" },
        absValue: { fontSize: "0.75rem", fontWeight: 600, color: t.textMuted },
        tabBar: { display: "flex", borderBottom: `1px solid ${t.divider}` },
        tabBtn: (active) => ({
            flex: 1,
            padding: "0.75rem",
            fontSize: "0.7rem",
            fontWeight: 700,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            border: "none",
            background: active
                ? (isDark ? "rgba(16,185,129,0.07)" : "rgba(5,150,105,0.05)")
                : "transparent",
            color: active ? t.brand : t.textDim,
            borderBottom: active ? `2px solid ${t.brand}` : "2px solid transparent",
            cursor: "pointer",
            transition: "all 0.2s ease",
        }),
        tabContent: { padding: "1.1rem" },
        copyBtn: {
            fontSize: "0.72rem",
            padding: "0.3rem 0.85rem",
            borderRadius: "8px",
            border: "none",
            background: t.brand,
            color: "#fff",
            cursor: "pointer",
            fontWeight: 600,
            transition: "opacity 0.2s",
        },
        extractedText: {
            fontSize: "0.875rem",
            color: t.text,
            whiteSpace: "pre-wrap",
            wordBreak: "break-words",
            lineHeight: 1.7,
            opacity: 0.88,
        },
        noiseSection: {
            marginTop: "1rem",
            paddingTop: "0.75rem",
            borderTop: `1px solid ${t.divider}`,
        },
        noiseSectionLabel: { fontSize: "0.65rem", color: t.textDim, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.4rem" },
        noiseItem: { fontSize: "0.75rem", color: t.textMuted },
        engineLabel: { marginTop: "0.75rem", fontSize: "0.65rem", color: t.textFaint },
        sectionLabel: { fontSize: "0.65rem", color: t.textDim, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.5rem" },
        entityTypeBadge: {
            fontSize: "0.62rem",
            color: t.brand,
            background: t.brandBg,
            border: `1px solid ${t.brandBorder}`,
            padding: "0.15rem 0.6rem",
            borderRadius: "999px",
        },
        pillBg: {
            background: t.pillBg,
            color: t.pillText,
            fontSize: "0.75rem",
            padding: "0.2rem 0.65rem",
            borderRadius: "999px",
            fontFamily: "monospace",
        },
        analysisBox: {
            background: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)",
            border: `1px solid ${t.divider}`,
            borderRadius: "12px",
            padding: "0.75rem",
        },
        analysisBoxLabel: { fontSize: "0.6rem", color: t.textDim, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "4px" },
        statGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" },
        statGridCell: {
            background: isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)",
            border: `1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)"}`,
            borderRadius: "10px",
            padding: "0.65rem",
        },
        statGridLabel: { fontSize: "0.6rem", color: t.textDim, marginBottom: "2px" },
        statGridValue: { fontSize: "0.875rem", color: t.text, fontWeight: 600, opacity: 0.88 },
        footer: {
            padding: "1.2rem 2.5rem",
            borderTop: `1px solid ${t.navBorder}`,
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
        },
        footerDot: { width: "6px", height: "6px", borderRadius: "50%", background: t.brand },
        footerText: { fontSize: "0.82rem", color: t.footerText },
    };

    return (
        <div style={styles.root}>
            <style>{`
                @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
                * { box-sizing: border-box; margin: 0; padding: 0; }
                @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.6;transform:scale(1.3)} }
                @keyframes spin { to{transform:rotate(360deg)} }
            `}</style>

            {/* Nav */}
            <nav style={styles.nav}>
                <div style={{ display: "flex", alignItems: "center" }}>
                    <div style={styles.brandDot} />
                    <span style={styles.brandText}>Text Extraction</span>
                </div>
                <button
                    onClick={() => setTheme(isDark ? "light" : "dark")}
                    style={styles.toggleBtn}
                    title={t.toggleLabel}
                >
                    <span style={{ fontSize: "0.85rem" }}>{t.toggleIcon}</span>
                    {t.toggleLabel}
                </button>
            </nav>

            {/* Main */}
            <main style={styles.main}>
                <div style={styles.gridOverlay} />
                <div style={styles.glow} />

                <div style={styles.content}>
                    <h1 className="anim-up" style={styles.h1}>
                        Text Extraction<br />
                        <span style={styles.h1Accent}>Scanned Docs & Images</span>
                    </h1>

                    <div className="anim-up" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem", marginTop: "2rem" }}>
                        <input type="file" accept="image/*" ref={fileInputRef} onChange={handleFileChange} style={{ display: "none" }} />

                        <button onClick={handleClick} style={styles.uploadBtn}
                            onMouseEnter={e => e.currentTarget.style.background = t.brandHover}
                            onMouseLeave={e => e.currentTarget.style.background = t.brand}>
                            Upload Image
                        </button>

                        {image && <img src={image} alt="Preview" style={styles.imgPreview} />}

                        {image && (
                            <div style={{ width: "100%", maxWidth: "560px" }}>
                                <label style={styles.gtLabel}>
                                    Ground Truth
                                    <span style={{ marginLeft: "0.5rem", textTransform: "none", color: t.textFaint }}>
                                        (optional — paste expected text to measure real accuracy)
                                    </span>
                                </label>
                                <textarea
                                    rows={3}
                                    value={groundTruth}
                                    onChange={e => setGroundTruth(e.target.value)}
                                    placeholder="Paste the correct/expected text here…"
                                    style={styles.gtTextarea}
                                />
                            </div>
                        )}

                        {image && (
                            <div style={{ display: "flex", gap: "0.6rem" }}>
                                <button onClick={handleFindText} disabled={loading} style={{ ...styles.actionBtn, opacity: loading ? 0.55 : 1 }}
                                    onMouseEnter={e => { if (!loading) e.currentTarget.style.background = t.brandHover; }}
                                    onMouseLeave={e => e.currentTarget.style.background = t.brand}>
                                    {loading ? "Processing…" : "Find Text"}
                                </button>
                                <button onClick={handleRemove} style={styles.removeBtn}
                                    onMouseEnter={e => e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.09)"}
                                    onMouseLeave={e => e.currentTarget.style.background = styles.removeBtn.background}>
                                    Remove
                                </button>
                            </div>
                        )}

                        {/* ── Result Card ── */}
                        {result && (
                            <div style={styles.card}>
                                {/* Stat bar */}
                                <div style={styles.statBar}>
                                    {[
                                        { label: displayAccuracyLabel, value: `${displayAccuracyScore}%`, customColor: accuracyColor, sub: displayAccuracyQuality, hint: absOverall !== null ? "CER/WER" : "estimated" },
                                        { label: "Confidence", value: `${confidence}%`, warn: isLowConf },
                                        { label: "Words", value: wordCount },
                                        { label: "Lines", value: lineCount },
                                        { label: "Type", value: docType || "—" },
                                    ].map(({ label, value, warn, customColor, sub, hint }, idx, arr) => (
                                        <div key={label} style={{ ...styles.statCell, borderRight: idx < arr.length - 1 ? `1px solid ${t.divider}` : "none" }}>
                                            <span style={styles.statLabel}>{label}</span>
                                            <span style={{ fontSize: "0.875rem", fontWeight: 700, color: customColor ? customColor : warn ? t.warnColor : t.okColor }}>
                                                {value}
                                            </span>
                                            {sub  && <span style={styles.statSub}>{sub}</span>}
                                            {hint && <span style={styles.statHint}>{hint}</span>}
                                        </div>
                                    ))}
                                </div>

                                {/* Absolute accuracy row */}
                                {absAccuracy && (
                                    <div style={styles.absRow}>
                                        {[
                                            { label: "Char Acc",  value: `${absAccuracy.char_accuracy}%` },
                                            { label: "Word Acc",  value: `${absAccuracy.word_accuracy}%` },
                                            { label: "CER",       value: absAccuracy.cer },
                                            { label: "WER",       value: absAccuracy.wer },
                                        ].map(({ label, value }, idx, arr) => (
                                            <div key={label} style={{ ...styles.absCell, borderRight: idx < arr.length - 1 ? `1px solid ${t.divider}` : "none" }}>
                                                <span style={styles.absLabel}>{label}</span>
                                                <span style={styles.absValue}>{value}</span>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* Tabs */}
                                <div style={styles.tabBar}>
                                    {tabs.map(tab => (
                                        <button key={tab} onClick={() => setActiveTab(tab)} style={styles.tabBtn(activeTab === tab)}>
                                            {tab}
                                        </button>
                                    ))}
                                </div>

                                {/* Tab: Text */}
                                {activeTab === "text" && (
                                    <div style={styles.tabContent}>
                                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                                            <span style={styles.sectionLabel}>Extracted text</span>
                                            <button onClick={handleCopy} style={styles.copyBtn}>
                                                {copied ? "Copied!" : "Copy"}
                                            </button>
                                        </div>
                                        <p style={styles.extractedText}>{cleanedText}</p>

                                        {noiseIssues.length > 0 && (
                                            <div style={styles.noiseSection}>
                                                <p style={styles.noiseSectionLabel}>Noise corrections applied</p>
                                                {noiseIssues.map((issue, i) => (
                                                    <p key={i} style={styles.noiseItem}>• {issue}</p>
                                                ))}
                                            </div>
                                        )}
                                        {engineUsed && <p style={styles.engineLabel}>Engine: {engineUsed}</p>}
                                    </div>
                                )}

                                {/* Tab: Entities */}
                                {activeTab === "entities" && (
                                    <div style={{ ...styles.tabContent, display: "flex", flexDirection: "column", gap: "1rem" }}>
                                        {Object.keys(namedEntities).length > 0 && (
                                            <div>
                                                <p style={styles.sectionLabel}>Named entities</p>
                                                {Object.entries(namedEntities).map(([type, values]) => (
                                                    <div key={type} style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginBottom: "0.5rem", alignItems: "center" }}>
                                                        <span style={{ fontSize: "0.6rem", color: t.textDim, textTransform: "uppercase", marginRight: "0.25rem" }}>{type}</span>
                                                        {values.map(v => (
                                                            <span key={v} style={styles.entityTypeBadge}>{v}</span>
                                                        ))}
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        {["file_paths","numeric_ids","emails","urls","currency_amounts","phone_numbers"].map(key => {
                                            const vals = regexEntities[key];
                                            if (!vals?.length) return null;
                                            return (
                                                <div key={key}>
                                                    <p style={styles.sectionLabel}>{key.replace(/_/g, " ")}</p>
                                                    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                                                        {vals.map((v, i) => <span key={i} style={styles.pillBg}>{v}</span>)}
                                                    </div>
                                                </div>
                                            );
                                        })}

                                        {Object.keys(namedEntities).length === 0 &&
                                            !["file_paths","numeric_ids","emails","urls","currency_amounts","phone_numbers"].some(k => regexEntities[k]?.length) && (
                                            <p style={{ fontSize: "0.875rem", color: t.textDim, padding: "1.5rem 0", textAlign: "center" }}>No entities detected</p>
                                        )}
                                    </div>
                                )}

                                {/* Tab: Analysis */}
                                {activeTab === "analysis" && (
                                    <div style={{ ...styles.tabContent, display: "flex", flexDirection: "column", gap: "1rem" }}>
                                        <div style={{ display: "flex", gap: "0.75rem" }}>
                                            <div style={{ ...styles.analysisBox, flex: 1 }}>
                                                <p style={styles.analysisBoxLabel}>Category</p>
                                                <p style={{ fontSize: "0.875rem", color: t.text, textTransform: "capitalize", opacity: 0.85 }}>{category || "—"}</p>
                                            </div>
                                            <div style={{ ...styles.analysisBox, flex: 1 }}>
                                                <p style={styles.analysisBoxLabel}>Sentiment</p>
                                                <p style={{ fontSize: "0.875rem", fontWeight: 700, textTransform: "capitalize", color: sentimentColor }}>{sentiment || "—"}</p>
                                            </div>
                                        </div>

                                        {keywords.length > 0 && (
                                            <div>
                                                <p style={styles.sectionLabel}>Top keywords</p>
                                                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                                                    {keywords.map(kw => (
                                                        <span key={kw} style={styles.entityTypeBadge}>{kw}</span>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        <div>
                                            <p style={styles.sectionLabel}>Document stats</p>
                                            <div style={styles.statGrid}>
                                                {[
                                                    { label: "Total words",     value: result?.semantic_analysis?.statistics?.total_words ?? wordCount },
                                                    { label: "Unique words",    value: result?.semantic_analysis?.statistics?.unique_words ?? "—" },
                                                    { label: "Total lines",     value: lineCount },
                                                    { label: "Avg line length", value: result?.semantic_analysis?.statistics?.avg_line_length?.toFixed(1) ?? "—" },
                                                ].map(({ label, value }) => (
                                                    <div key={label} style={styles.statGridCell}>
                                                        <p style={styles.statGridLabel}>{label}</p>
                                                        <p style={styles.statGridValue}>{value}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer style={styles.footer}>
                <div style={styles.footerDot} />
                <span style={styles.footerText}>Text Extraction</span>
            </footer>
        </div>
    );
}