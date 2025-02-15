import React, { createContext, useContext, useState, useEffect } from "react";
import axios, { AxiosError } from "axios";

// Types
export type AnalysisType =
  | "filter"
  | "frameworker"
  | "classifier"
  | "wizard"
  | "comparative";

export interface FileInfo {
  name: string;
  path: string;
  size: number;
  modified: number;
}

// Define result types for each analysis type
export interface FilterResult {
  total: number;
  filtered: number;
  data: Array<{
    "Post/comments": string;
    Platform: string;
    Sentiment: string;
    similarity_score?: number;
  }>;
}

export interface FrameworkerResult {
  classes: Array<{
    name: string;
    description: string;
    keywords: string[];
    examples: string[];
  }>;
}

export interface ClassifierResult {
  summary: {
    total_texts: number;
    class_distribution: Record<string, number>;
  };
  [className: string]: any;
}

export interface WizardResult {
  insights: Array<{
    observation: string;
    evidence: string[];
    implications: string;
    confidence: string;
  }>;
  summary: string;
  recommendations: string[];
}

export interface ComparativeResult {
  comparative_insights: Array<{
    aspect: string;
    dataset1_position?: string;
    dataset2_position?: string;
    competitor1_position?: string;
    competitor2_position?: string;
    key_differences: string;
    implications: string;
  }>;
  summary: string;
  metric_comparisons?: {
    distributions: {
      dataset1: string;
      dataset2: string;
      differences: string;
    };
    patterns: {
      common_patterns: string[];
      unique_to_dataset1: string[];
      unique_to_dataset2: string[];
    };
  };
  key_metrics_comparison?: {
    sentiment: {
      competitor1: string;
      competitor2: string;
      difference: string;
    };
    user_satisfaction: {
      competitor1_score: string;
      competitor2_score: string;
      analysis: string;
    };
  };
  key_findings?: {
    similarities: string[];
    differences: string[];
  };
  competitive_advantages?: {
    competitor1: string[];
    competitor2: string[];
  };
  recommendations:
    | string[]
    | {
        competitor1: string[];
        competitor2: string[];
      };
}

export type AnalysisResult =
  | FilterResult
  | FrameworkerResult
  | ClassifierResult
  | WizardResult
  | ComparativeResult;

export interface AnalysisState {
  files: FileInfo[];
  selectedFile: string;
  selectedFile2: string; // For comparative analysis
  analysisType: AnalysisType;
  useSemanticFilter: boolean;
  targetClasses?: number;
  classNames: string[];
  isMultiClass: boolean;
  prompt: string;
  description1: string;
  description2: string;
  isCompetitive: boolean;
  loading: boolean;
  error: string | null;
  results: AnalysisResult | null;
}

interface AnalysisContextType {
  state: AnalysisState;
  setState: React.Dispatch<React.SetStateAction<AnalysisState>>;
  runAnalysis: () => Promise<void>;
  uploadFile: (file: File) => Promise<void>;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(
  undefined
);

const API_BASE_URL = "http://localhost:8000/api";

export function AnalysisProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AnalysisState>({
    files: [],
    selectedFile: "",
    selectedFile2: "",
    analysisType: "filter",
    useSemanticFilter: true,
    classNames: [],
    isMultiClass: false,
    prompt: "",
    description1: "",
    description2: "",
    isCompetitive: false,
    loading: false,
    error: null,
    results: null,
  });

  // Fetch available files on mount
  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/files`);
      setState((prev) => ({ ...prev, files: response.data.files }));
    } catch (err) {
      const error = err as AxiosError;
      setState((prev) => ({
        ...prev,
        error: error.response?.data?.detail || "Failed to fetch files",
      }));
    }
  };

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      setState((prev) => ({ ...prev, loading: true, error: null }));
      await axios.post(`${API_BASE_URL}/upload`, formData);
      await fetchFiles(); // Refresh file list
    } catch (err) {
      const error = err as AxiosError;
      setState((prev) => ({
        ...prev,
        error: error.response?.data?.detail || "Failed to upload file",
      }));
    } finally {
      setState((prev) => ({ ...prev, loading: false }));
    }
  };

  const runAnalysis = async () => {
    setState((prev) => ({
      ...prev,
      loading: true,
      error: null,
      results: null,
    }));

    try {
      let response;

      switch (state.analysisType) {
        case "filter":
          response = await axios.post(`${API_BASE_URL}/filter`, {
            file: state.selectedFile,
            criteria: state.prompt,
            use_semantic: state.useSemanticFilter,
          });
          break;

        case "frameworker":
          response = await axios.post(`${API_BASE_URL}/frameworker`, {
            file: state.selectedFile,
            target_classes: state.targetClasses,
            context: state.prompt,
          });
          break;

        case "classifier":
          response = await axios.post(`${API_BASE_URL}/classifier`, {
            file: state.selectedFile,
            classes: state.classNames,
            is_multi_class: state.isMultiClass,
          });
          break;

        case "wizard":
          response = await axios.post(`${API_BASE_URL}/wizard`, {
            file: state.selectedFile,
            question: state.prompt,
            use_semantic: state.useSemanticFilter,
          });
          break;

        case "comparative":
          response = await axios.post(`${API_BASE_URL}/comparative`, {
            file1: state.selectedFile,
            file2: state.selectedFile2,
            description1: state.description1,
            description2: state.description2,
            question: state.prompt,
            use_semantic: state.useSemanticFilter,
            is_competitive: state.isCompetitive,
          });
          break;
      }

      setState((prev) => ({ ...prev, results: response?.data }));
    } catch (err) {
      const error = err as AxiosError;
      setState((prev) => ({
        ...prev,
        error:
          error.response?.data?.detail || "An error occurred during analysis",
      }));
    } finally {
      setState((prev) => ({ ...prev, loading: false }));
    }
  };

  return (
    <AnalysisContext.Provider
      value={{ state, setState, runAnalysis, uploadFile }}
    >
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis() {
  const context = useContext(AnalysisContext);
  if (context === undefined) {
    throw new Error("useAnalysis must be used within an AnalysisProvider");
  }
  return context;
}
