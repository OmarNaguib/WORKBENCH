import React from "react";
import {
  Form,
  Select,
  Input,
  InputNumber,
  Switch,
  Button,
  Space,
  Upload,
  message,
} from "antd";
import { UploadOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import { useAnalysis, AnalysisType } from "../context/AnalysisContext";

const { Option } = Select;
const { TextArea } = Input;

const AnalysisForm: React.FC = () => {
  const navigate = useNavigate();
  const { state, setState, runAnalysis, uploadFile } = useAnalysis();

  const handleAnalysisTypeChange = (value: AnalysisType) => {
    setState((prev) => ({ ...prev, analysisType: value }));
  };

  const handleFileUpload = async (file: File) => {
    try {
      await uploadFile(file);
      message.success("File uploaded successfully");
      return false; // Prevent default upload behavior
    } catch {
      message.error("Failed to upload file");
      return false;
    }
  };

  const handleSubmit = async () => {
    try {
      await runAnalysis();
      // Navigate to results page
      navigate(`/results/${state.analysisType}/${Date.now()}`);
    } catch {
      message.error("Failed to run analysis");
    }
  };

  return (
    <Form layout="vertical" onFinish={handleSubmit}>
      <Space direction="vertical" style={{ width: "100%" }} size="large">
        {/* File Upload */}
        <Upload
          accept=".xlsx"
          beforeUpload={handleFileUpload}
          showUploadList={false}
        >
          <Button icon={<UploadOutlined />}>Upload Excel File</Button>
        </Upload>

        {/* Analysis Type Selection */}
        <Form.Item label="Analysis Type">
          <Select
            value={state.analysisType}
            onChange={handleAnalysisTypeChange}
          >
            <Option value="filter">Filter</Option>
            <Option value="frameworker">Frameworker</Option>
            <Option value="classifier">Classifier</Option>
            <Option value="wizard">Data Wizard</Option>
            <Option value="comparative">Comparative Analysis</Option>
          </Select>
        </Form.Item>

        {/* File Selection */}
        <Form.Item label="Select Data File">
          <Select
            value={state.selectedFile}
            onChange={(value) =>
              setState((prev) => ({ ...prev, selectedFile: value }))
            }
          >
            {state.files.map((file) => (
              <Option key={file.name} value={file.name}>
                {file.name}
              </Option>
            ))}
          </Select>
        </Form.Item>

        {/* Analysis Type Specific Fields */}
        {state.analysisType === "filter" && (
          <Form.Item label="Filter Type">
            <Switch
              checked={state.useSemanticFilter}
              onChange={(checked) =>
                setState((prev) => ({ ...prev, useSemanticFilter: checked }))
              }
              checkedChildren="Semantic"
              unCheckedChildren="LLM"
            />
          </Form.Item>
        )}

        {state.analysisType === "wizard" && (
          <Form.Item label="Filter Type">
            <Switch
              checked={state.useSemanticFilter}
              onChange={(checked) =>
                setState((prev) => ({ ...prev, useSemanticFilter: checked }))
              }
              checkedChildren="Semantic"
              unCheckedChildren="LLM"
            />
          </Form.Item>
        )}

        {state.analysisType === "frameworker" && (
          <Form.Item label="Target Number of Classes (Optional)">
            <InputNumber
              value={state.targetClasses}
              onChange={(value) =>
                setState((prev) => ({
                  ...prev,
                  targetClasses: value || undefined,
                }))
              }
              min={1}
            />
          </Form.Item>
        )}

        {state.analysisType === "classifier" && (
          <>
            <Form.Item label="Class Names (Comma-separated)">
              <Input
                value={state.classNames.join(", ")}
                onChange={(e) =>
                  setState((prev) => ({
                    ...prev,
                    classNames: e.target.value
                      .split(",")
                      .map((s) => s.trim())
                      .filter(Boolean),
                  }))
                }
                placeholder="e.g., Technical Issues, Customer Service, Product Feedback"
              />
            </Form.Item>
            <Form.Item label="Multi-class Classification">
              <Switch
                checked={state.isMultiClass}
                onChange={(checked) =>
                  setState((prev) => ({ ...prev, isMultiClass: checked }))
                }
              />
            </Form.Item>
          </>
        )}

        {state.analysisType === "comparative" && (
          <>
            <Form.Item label="Second Data File">
              <Select
                value={state.selectedFile2}
                onChange={(value) =>
                  setState((prev) => ({ ...prev, selectedFile2: value }))
                }
              >
                {state.files.map((file) => (
                  <Option key={file.name} value={file.name}>
                    {file.name}
                  </Option>
                ))}
              </Select>
            </Form.Item>
            <Form.Item label="Dataset 1 Description">
              <Input
                value={state.description1}
                onChange={(e) =>
                  setState((prev) => ({
                    ...prev,
                    description1: e.target.value,
                  }))
                }
                placeholder="e.g., Customer feedback for Service A"
              />
            </Form.Item>
            <Form.Item label="Dataset 2 Description">
              <Input
                value={state.description2}
                onChange={(e) =>
                  setState((prev) => ({
                    ...prev,
                    description2: e.target.value,
                  }))
                }
                placeholder="e.g., Customer feedback for Service B"
              />
            </Form.Item>
            <Form.Item label="Filter Type">
              <Switch
                checked={state.useSemanticFilter}
                onChange={(checked) =>
                  setState((prev) => ({ ...prev, useSemanticFilter: checked }))
                }
                checkedChildren="Semantic"
                unCheckedChildren="LLM"
              />
            </Form.Item>
            <Form.Item label="Competitive Analysis">
              <Switch
                checked={state.isCompetitive}
                onChange={(checked) =>
                  setState((prev) => ({ ...prev, isCompetitive: checked }))
                }
              />
            </Form.Item>
          </>
        )}

        {/* Prompt/Question Input */}
        <Form.Item
          label={
            state.analysisType === "wizard" ? "Question" : "Prompt/Context"
          }
        >
          <TextArea
            value={state.prompt}
            onChange={(e) =>
              setState((prev) => ({ ...prev, prompt: e.target.value }))
            }
            rows={4}
            placeholder={
              state.analysisType === "wizard"
                ? "Enter your question about the data..."
                : "Enter prompt or context for analysis..."
            }
          />
        </Form.Item>

        {/* Submit Button */}
        <Form.Item>
          <Button
            type="primary"
            htmlType="submit"
            loading={state.loading}
            disabled={!state.selectedFile || !state.prompt}
          >
            Run Analysis
          </Button>
        </Form.Item>
      </Space>
    </Form>
  );
};

export default AnalysisForm;
