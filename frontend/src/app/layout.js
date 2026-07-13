import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata = {
  title: "Resume Ranker — Match Your Resume to Any Job",
  description:
    "Score how well your resume matches a job description using NLP-powered TF-IDF analysis and cosine similarity.",
  keywords: ["resume", "ranker", "NLP", "job description", "cosine similarity", "TF-IDF"],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" data-theme="light">
      <body className={inter.variable}>{children}</body>
    </html>
  );
}
