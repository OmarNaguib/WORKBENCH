import React from "react";
import { Card, Typography, Row, Col, Tag, Empty, List } from "antd";
import { useAnalysis } from "../../context/AnalysisContext";
import { FrameworkerResult } from "../../context/AnalysisContext";

const { Title, Text, Paragraph } = Typography;

const FrameworkerResults: React.FC = () => {
  const { state } = useAnalysis();
  const results = state.results as FrameworkerResult;

  if (!results) {
    return <Empty description="No results available" />;
  }

  return (
    <div>
      <Title level={3}>Discovered Classes</Title>
      <Text>Found {results.classes.length} distinct classes</Text>

      <Row gutter={[16, 16]} style={{ marginTop: "24px" }}>
        {results.classes.map((classInfo, index) => (
          <Col xs={24} md={12} key={index}>
            <Card title={classInfo.name}>
              <Paragraph>{classInfo.description}</Paragraph>

              <div style={{ marginBottom: "12px" }}>
                <Text strong>Keywords:</Text>
                <div style={{ marginTop: "8px" }}>
                  {classInfo.keywords.map((keyword, idx) => (
                    <Tag key={idx} color="blue" style={{ marginBottom: "4px" }}>
                      {keyword}
                    </Tag>
                  ))}
                </div>
              </div>

              <div>
                <Text strong>Example Comments:</Text>
                <List
                  size="small"
                  dataSource={classInfo.examples}
                  renderItem={(example) => (
                    <List.Item>
                      <Text>{example}</Text>
                    </List.Item>
                  )}
                  style={{ marginTop: "8px" }}
                />
              </div>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};

export default FrameworkerResults;
