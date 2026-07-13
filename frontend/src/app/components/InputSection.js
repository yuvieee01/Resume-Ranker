"use client";

import { useState, useRef } from "react";
import styles from "./InputSection.module.css";

export default function InputSection({ label, id, onDataChange }) {
  const [mode, setMode] = useState("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const fileRef = useRef(null);

  function handleModeSwitch(newMode) {
    setMode(newMode);
    if (newMode === "text") {
      setFile(null);
      onDataChange({ type: "text", value: text });
    } else {
      setText("");
      onDataChange({ type: "file", value: file });
    }
  }

  function handleTextChange(e) {
    setText(e.target.value);
    onDataChange({ type: "text", value: e.target.value });
  }

  function handleFileChange(e) {
    const selected = e.target.files[0];
    if (selected) {
      const ext = selected.name.split(".").pop().toLowerCase();
      if (ext !== "pdf" && ext !== "txt") {
        alert("Only PDF and TXT files are supported.");
        e.target.value = "";
        return;
      }
      setFile(selected);
      onDataChange({ type: "file", value: selected });
    }
  }

  function clearFile() {
    setFile(null);
    if (fileRef.current) fileRef.current.value = "";
    onDataChange({ type: "file", value: null });
  }

  return (
    <div className={styles.section}>
      <div className={styles.header}>
        <label className={styles.label} htmlFor={id}>
          {label}
        </label>
        <div className={styles.modeSwitch}>
          <button
            type="button"
            className={`${styles.modeBtn} ${mode === "text" ? styles.active : ""}`}
            onClick={() => handleModeSwitch("text")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
            Text
          </button>
          <button
            type="button"
            className={`${styles.modeBtn} ${mode === "file" ? styles.active : ""}`}
            onClick={() => handleModeSwitch("file")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            PDF
          </button>
        </div>
      </div>

      {mode === "text" ? (
        <textarea
          id={id}
          className={styles.textarea}
          placeholder={`Paste your ${label.toLowerCase()} here...`}
          value={text}
          onChange={handleTextChange}
          rows={7}
        />
      ) : (
        <div className={styles.dropzone}>
          {file ? (
            <div className={styles.filePreview}>
              <div className={styles.fileIcon}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                </svg>
              </div>
              <div className={styles.fileInfo}>
                <span className={styles.fileName}>{file.name}</span>
                <span className={styles.fileSize}>
                  {(file.size / 1024).toFixed(1)} KB
                </span>
              </div>
              <button
                type="button"
                className={styles.removeBtn}
                onClick={clearFile}
                aria-label="Remove file"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          ) : (
            <label className={styles.dropLabel} htmlFor={`${id}-file`}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <span className={styles.dropText}>
                Click to upload a <strong>PDF</strong> or <strong>TXT</strong> file
              </span>
              <span className={styles.dropHint}>Max 10 MB</span>
            </label>
          )}
          <input
            ref={fileRef}
            id={`${id}-file`}
            type="file"
            accept=".pdf,.txt"
            className={styles.fileInput}
            onChange={handleFileChange}
          />
        </div>
      )}
    </div>
  );
}
