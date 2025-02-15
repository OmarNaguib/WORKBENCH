import React from "react";
import { Card, Typography, Row, Col, Tag, Empty, List, Divider } from "antd";
import { useAnalysis } from "../../context/AnalysisContext";
import { WizardResult } from "../../context/AnalysisContext";

const { Title, Text, Paragraph } = Typography;

const WizardResults: React.FC = () => {
  const { state } = useAnalysis();
  const results = state.results as WizardResult;

  if (!results) {
    return <Empty description="No results available" />;
  }

  return (
    <div>
      <Title level={3}>Analysis Results</Title>

      {/* Summary */}
      <Card style={{ marginBottom: "24px" }}>
        <Title level={4}>Summary</Title>
        <Paragraph>{results.summary}</Paragraph>
      </Card>

      {/* Key Insights */}
      <Title level={4}>Key Insights</Title>
      <Row gutter={[16, 16]}>
        {results.insights.map((insight, index) => (
          <Col xs={24} md={12} key={index}>
            <Card>
              <Title level={5}>{insight.observation}</Title>
              <Paragraph>{insight.implications}</Paragraph>

              <div style={{ marginBottom: "12px" }}>
                <Tag
                  color={
                    insight.confidence === "High"
                      ? "green"
                      : insight.confidence === "Medium"
                      ? "orange"
                      : "red"
                  }
                >
                  Confidence: {insight.confidence}
                </Tag>
              </div>

              <Text strong>Supporting Evidence:</Text>
              <List
                size="small"
                dataSource={insight.evidence}
                renderItem={(evidence) => (
                  <List.Item>
                    <Text>{evidence}</Text>
                  </List.Item>
                )}
                style={{ marginTop: "8px" }}
              />
            </Card>
          </Col>
        ))}
      </Row>

      <Divider />

      {/* Recommendations */}
      <Title level={4}>Recommendations</Title>
      <Card>
        <List
          dataSource={results.recommendations}
          renderItem={(recommendation, index) => (
            <List.Item>
              <Text>
                {index + 1}. {recommendation}
              </Text>
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
};

export default WizardResults;
