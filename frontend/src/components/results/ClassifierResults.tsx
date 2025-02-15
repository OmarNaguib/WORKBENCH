import React from "react";
import { Card, Typography, Row, Col, Tag, Empty, List, Divider } from "antd";
import { useAnalysis } from "../../context/AnalysisContext";
import { ClassifierResult } from "../../context/AnalysisContext";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Doughnut } from "react-chartjs-2";

ChartJS.register(ArcElement, Tooltip, Legend);

const { Title, Text } = Typography;

interface ClassifiedItem {
  text: string;
  confidence?: number;
}

const ClassifierResults: React.FC = () => {
  const { state } = useAnalysis();
  const results = state.results as ClassifierResult;

  if (!results) {
    return <Empty description="No results available" />;
  }

  // Prepare data for the donut chart
  const chartData = {
    labels: Object.keys(results.summary.class_distribution).filter(
      (key) => key !== "Other"
    ),
    datasets: [
      {
        data: Object.entries(results.summary.class_distribution)
          .filter(([key]) => key !== "Other")
          .map(([, value]) => value),
        backgroundColor: [
          "#FF6384",
          "#36A2EB",
          "#FFCE56",
          "#4BC0C0",
          "#9966FF",
          "#FF9F40",
        ],
        hoverBackgroundColor: [
          "#FF6384",
          "#36A2EB",
          "#FFCE56",
          "#4BC0C0",
          "#9966FF",
          "#FF9F40",
        ],
      },
    ],
  };

  return (
    <div>
      <Title level={3}>Classification Results</Title>
      <Text>
        Analyzed {results.summary.total_texts} texts across{" "}
        {Object.keys(results.summary.class_distribution).length} classes
      </Text>

      {/* Distribution Chart */}
      <Row gutter={[16, 16]} style={{ marginTop: "24px" }}>
        <Col xs={24} md={12}>
          <Card title="Class Distribution">
            <div style={{ height: "300px", position: "relative" }}>
              <Doughnut
                data={chartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                }}
              />
            </div>
          </Card>
        </Col>
        <Col xs={24} md={12}>
          <Card title="Distribution Summary">
            <List
              dataSource={Object.entries(results.summary.class_distribution)}
              renderItem={([className, count]) => (
                <List.Item>
                  <Text>
                    {className}:{" "}
                    <Tag color="blue">
                      {count} texts (
                      {((count / results.summary.total_texts) * 100).toFixed(1)}
                      %)
                    </Tag>
                  </Text>
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      <Divider />

      {/* Class Details */}
      <Title level={4}>Class Details</Title>
      <Row gutter={[16, 16]}>
        {Object.entries(results)
          .filter(([key]) => key !== "summary")
          .map(([className, items]) => (
            <Col xs={24} md={12} key={className}>
              <Card
                title={
                  <span>
                    {className}{" "}
                    <Tag color="blue">
                      {items.length} texts (
                      {(
                        (items.length / results.summary.total_texts) *
                        100
                      ).toFixed(1)}
                      %)
                    </Tag>
                  </span>
                }
              >
                <List
                  size="small"
                  dataSource={items.slice(0, 5)} // Show only first 5 examples
                  renderItem={(item: ClassifiedItem) => (
                    <List.Item>
                      <Text>
                        {item.text}
                        {item.confidence && (
                          <Tag color="purple" style={{ marginLeft: 8 }}>
                            Confidence: {(item.confidence * 100).toFixed(1)}%
                          </Tag>
                        )}
                      </Text>
                    </List.Item>
                  )}
                  footer={
                    items.length > 5 && (
                      <div style={{ textAlign: "center" }}>
                        <Text type="secondary">
                          And {items.length - 5} more items...
                        </Text>
                      </div>
                    )
                  }
                />
              </Card>
            </Col>
          ))}
      </Row>
    </div>
  );
};

export default ClassifierResults;
