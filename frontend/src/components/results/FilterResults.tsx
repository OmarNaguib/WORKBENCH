import React from "react";
import { Card, Typography, Row, Col, Tag, Empty } from "antd";
import { useAnalysis } from "../../context/AnalysisContext";
import { FilterResult } from "../../context/AnalysisContext";

const { Title, Text } = Typography;

const FilterResults: React.FC = () => {
  const { state } = useAnalysis();
  const results = state.results as FilterResult;

  if (!results) {
    return <Empty description="No results available" />;
  }

  return (
    <div>
      <Title level={3}>Filter Results</Title>
      <Text>
        Found {results.filtered} matching items out of {results.total} total
      </Text>

      <Row gutter={[16, 16]} style={{ marginTop: "24px" }}>
        {results.data.map((item, index) => (
          <Col xs={24} sm={12} md={8} key={index}>
            <Card>
              <Text>{item["Post/comments"]}</Text>
              <div style={{ marginTop: "12px" }}>
                <Tag color="blue">{item.Platform}</Tag>
                <Tag
                  color={
                    item.Sentiment === "positive"
                      ? "green"
                      : item.Sentiment === "negative"
                      ? "red"
                      : "default"
                  }
                >
                  {item.Sentiment}
                </Tag>
                {item.similarity_score !== undefined && (
                  <Tag color="purple">
                    Similarity: {(item.similarity_score * 100).toFixed(1)}%
                  </Tag>
                )}
              </div>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};

export default FilterResults;
