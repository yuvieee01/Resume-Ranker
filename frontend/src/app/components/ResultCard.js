"use client";

import { useEffect, useState } from "react";
import styles from "./ResultCard.module.css";

export default function ResultCard({ score, percentage, jdLength, resumeLength }) {
  const [animated, setAnimated] = useState(0);

  useEffect(() => {
    // animate the gauge from 0 to the actual percentage
    let frame;
    const duration = 800;
    const start = performance.now();

    function step(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setAnimated(eased * percentage);
      if (progress < 1) frame = requestAnimationFrame(step);
    }

    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [percentage]);

  function getGrade(pct) {
    if (pct >= 75) return { label: "Excellent Match", color: "var(--success)", bg: "var(--success-light)" };
    if (pct >= 50) return { label: "Good Match", color: "var(--accent)", bg: "var(--accent-light)" };
    if (pct >= 25) return { label: "Fair Match", color: "var(--warning)", bg: "var(--warning-light)" };
    return { label: "Low Match", color: "var(--error)", bg: "var(--error-light)" };
  }

  const grade = getGrade(percentage);

  // SVG gauge dimensions
  const radius = 70;
  const stroke = 8;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (animated / 100) * circumference;

  return (
    <div className={styles.card}>
      <div className={styles.gaugeWrap}>
        <svg
          className={styles.gauge}
          width="160"
          height="160"
          viewBox="0 0 160 160"
        >
          <circle
            className={styles.track}
            cx="80"
            cy="80"
            r={radius}
            strokeWidth={stroke}
            fill="none"
          />
          <circle
            className={styles.fill}
            cx="80"
            cy="80"
            r={radius}
            strokeWidth={stroke}
            fill="none"
            style={{
              strokeDasharray: circumference,
              strokeDashoffset: offset,
              stroke: grade.color,
            }}
          />
        </svg>
        <div className={styles.gaugeText}>
          <span className={styles.scoreNum}>{Math.round(animated)}</span>
          <span className={styles.scorePercent}>%</span>
        </div>
      </div>

      <div className={styles.details}>
        <span
          className={styles.badge}
          style={{ background: grade.bg, color: grade.color }}
        >
          {grade.label}
        </span>

        <div className={styles.meta}>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>Cosine Score</span>
            <span className={styles.metaValue}>{score}</span>
          </div>
          <div className={styles.divider} />
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>JD Length</span>
            <span className={styles.metaValue}>{jdLength.toLocaleString()} chars</span>
          </div>
          <div className={styles.divider} />
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>Resume Length</span>
            <span className={styles.metaValue}>{resumeLength.toLocaleString()} chars</span>
          </div>
        </div>
      </div>
    </div>
  );
}
