"use client";

import { useState } from "react";
import styles from "./page.module.css";
import ThemeToggle from "./components/ThemeToggle";
import InputSection from "./components/InputSection";
import ResultCard from "./components/ResultCard";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [jdData, setJdData] = useState({ type: "text", value: "" });
  const [resumeData, setResumeData] = useState({ type: "text", value: "" });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setResult(null);

    // validate inputs
    const hasJd =
      (jdData.type === "text" && jdData.value?.trim()) ||
      (jdData.type === "file" && jdData.value);
    const hasResume =
      (resumeData.type === "text" && resumeData.value?.trim()) ||
      (resumeData.type === "file" && resumeData.value);

    if (!hasJd) {
      setError("Please provide a job description.");
      return;
    }
    if (!hasResume) {
      setError("Please provide a resume.");
      return;
    }

    setLoading(true);

    try {
      const body = new FormData();

      if (jdData.type === "text") {
        body.append("jd_text", jdData.value);
      } else {
        body.append("jd_file", jdData.value);
      }

      if (resumeData.type === "text") {
        body.append("resume_text", resumeData.value);
      } else {
        body.append("resume_file", resumeData.value);
      }

      const res = await fetch(`${API_URL}/api/rank`, {
        method: "POST",
        body,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.page}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.brand}>
            <div className={styles.logo}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
            </div>
            <div>
              <h1 className={styles.title}>Resume Ranker</h1>
              <p className={styles.subtitle}>NLP-powered resume scoring</p>
            </div>
          </div>
          <ThemeToggle />
        </div>
      </header>

      {/* Main */}
      <main className={styles.main}>
        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.inputGrid}>
            <InputSection
              label="Job Description"
              id="jd-input"
              onDataChange={setJdData}
            />
            <InputSection
              label="Resume"
              id="resume-input"
              onDataChange={setResumeData}
            />
          </div>

          {error && (
            <div className={styles.error}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              {error}
            </div>
          )}

          <button
            type="submit"
            className={styles.submitBtn}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className={styles.spinner} />
                Analyzing...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
                Analyze Match
              </>
            )}
          </button>
        </form>

        {result && (
          <section className={styles.results}>
            <ResultCard
              score={result.score}
              percentage={result.percentage}
              jdLength={result.jd_length}
              resumeLength={result.resume_length}
            />
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className={styles.footer}>
        <p>
          Built with <strong>spaCy</strong>, <strong>scikit-learn</strong> &{" "}
          <strong>Next.js</strong>
        </p>
      </footer>
    </div>
  );
}
