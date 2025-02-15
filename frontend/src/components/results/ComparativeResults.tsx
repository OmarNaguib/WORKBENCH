import React from "react";
import {
  Card,
  Typography,
  Row,
  Col,
  Tag,
  Empty,
  List,
  Divider,
  Tabs,
} from "antd";
import { useAnalysis } from "../../context/AnalysisContext";
import { ComparativeResult } from "../../context/AnalysisContext";

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

const ComparativeResults: React.FC = () => {
  const { state } = useAnalysis();
  const results = state.results as ComparativeResult;

  if (!results || !results.comparative_insights) {
    return <Empty description="No results available" />;
  }

  const isCompetitive = "competitive_advantages" in results;

  return (
    <div>
      <Title level={3}>
        {isCompetitive ? "Competitive Analysis" : "Comparative Analysis"}
      </Title>

      {/* Summary */}
      {results.summary && (
        <Card style={{ marginBottom: "24px" }}>
          <Title level={4}>Summary</Title>
          <Paragraph>{results.summary}</Paragraph>
        </Card>
      )}

      {/* Comparative Insights */}
      {results.comparative_insights.length > 0 && (
        <>
          <Title level={4}>Key Insights</Title>
          <Row gutter={[16, 16]}>
            {results.comparative_insights.map((insight, index) => (
              <Col xs={24} key={index}>
                <Card title={insight.aspect}>
                  <Row gutter={16}>
                    <Col xs={24} md={12}>
                      <Title level={5}>
                        {isCompetitive ? "Competitor 1" : "Dataset 1"}
                      </Title>
                      <Paragraph>
                        {isCompetitive
                          ? insight.competitor1_position
                          : insight.dataset1_position}
                      </Paragraph>
                    </Col>
                    <Col xs={24} md={12}>
                      <Title level={5}>
                        {isCompetitive ? "Competitor 2" : "Dataset 2"}
                      </Title>
                      <Paragraph>
                        {isCompetitive
                          ? insight.competitor2_position
                          : insight.dataset2_position}
                      </Paragraph>
                    </Col>
                  </Row>
                  <Divider />
                  <Title level={5}>Key Differences</Title>
                  <Paragraph>{insight.key_differences}</Paragraph>
                  <Title level={5}>Implications</Title>
                  <Paragraph>{insight.implications}</Paragraph>
                </Card>
              </Col>
            ))}
          </Row>
        </>
      )}

      {/* Metrics Comparison */}
      {results.metric_comparisons && (
        <>
          <Divider />
          <Title level={4}>Metric Comparisons</Title>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Card title="Distribution Analysis">
                <Paragraph>
                  {results.metric_comparisons.distributions.dataset1}
                </Paragraph>
                <Paragraph>
                  {results.metric_comparisons.distributions.dataset2}
                </Paragraph>
                <Divider />
                <Text strong>Key Differences:</Text>
                <Paragraph>
                  {results.metric_comparisons.distributions.differences}
                </Paragraph>
              </Card>
            </Col>
            <Col xs={24} md={12}>
              <Card title="Pattern Analysis">
                <Title level={5}>Common Patterns</Title>
                <List
                  size="small"
                  dataSource={
                    results.metric_comparisons.patterns.common_patterns
                  }
                  renderItem={(pattern) => <List.Item>{pattern}</List.Item>}
                />
                <Divider />
                <Tabs>
                  <TabPane tab="Unique to Dataset 1" key="1">
                    <List
                      size="small"
                      dataSource={
                        results.metric_comparisons.patterns.unique_to_dataset1
                      }
                      renderItem={(pattern) => <List.Item>{pattern}</List.Item>}
                    />
                  </TabPane>
                  <TabPane tab="Unique to Dataset 2" key="2">
                    <List
                      size="small"
                      dataSource={
                        results.metric_comparisons.patterns.unique_to_dataset2
                      }
                      renderItem={(pattern) => <List.Item>{pattern}</List.Item>}
                    />
                  </TabPane>
                </Tabs>
              </Card>
            </Col>
          </Row>
        </>
      )}

      {/* Competitive Metrics */}
      {results.key_metrics_comparison && (
        <>
          <Divider />
          <Title level={4}>Key Metrics</Title>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Card title="Sentiment Analysis">
                <Paragraph>
                  {results.key_metrics_comparison.sentiment.competitor1}
                </Paragraph>
                <Paragraph>
                  {results.key_metrics_comparison.sentiment.competitor2}
                </Paragraph>
                <Divider />
                <Text strong>Key Differences:</Text>
                <Paragraph>
                  {results.key_metrics_comparison.sentiment.difference}
                </Paragraph>
              </Card>
            </Col>
            <Col xs={24} md={12}>
              <Card title="User Satisfaction">
                <Row gutter={16}>
                  <Col span={12}>
                    <Title level={5}>Competitor 1</Title>
                    <Tag color="blue">
                      {
                        results.key_metrics_comparison.user_satisfaction
                          .competitor1_score
                      }
                    </Tag>
                  </Col>
                  <Col span={12}>
                    <Title level={5}>Competitor 2</Title>
                    <Tag color="blue">
                      {
                        results.key_metrics_comparison.user_satisfaction
                          .competitor2_score
                      }
                    </Tag>
                  </Col>
                </Row>
                <Divider />
                <Paragraph>
                  {results.key_metrics_comparison.user_satisfaction.analysis}
                </Paragraph>
              </Card>
            </Col>
          </Row>
        </>
      )}

      {/* Competitive Advantages */}
      {results.competitive_advantages && (
        <>
          <Divider />
          <Title level={4}>Competitive Advantages</Title>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Card title="Competitor 1">
                <List
                  size="small"
                  dataSource={results.competitive_advantages.competitor1}
                  renderItem={(advantage) => <List.Item>{advantage}</List.Item>}
                />
              </Card>
            </Col>
            <Col xs={24} md={12}>
              <Card title="Competitor 2">
                <List
                  size="small"
                  dataSource={results.competitive_advantages.competitor2}
                  renderItem={(advantage) => <List.Item>{advantage}</List.Item>}
                />
              </Card>
            </Col>
          </Row>
        </>
      )}

      {/* Recommendations */}
      {results.recommendations && (
        <>
          <Divider />
          <Title level={4}>Recommendations</Title>
          {Array.isArray(results.recommendations) ? (
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
          ) : (
            <Row gutter={[16, 16]}>
              <Col xs={24} md={12}>
                <Card title="For Competitor 1">
                  <List
                    dataSource={results.recommendations.competitor1}
                    renderItem={(recommendation, index) => (
                      <List.Item>
                        <Text>
                          {index + 1}. {recommendation}
                        </Text>
                      </List.Item>
                    )}
                  />
                </Card>
              </Col>
              <Col xs={24} md={12}>
                <Card title="For Competitor 2">
                  <List
                    dataSource={results.recommendations.competitor2}
                    renderItem={(recommendation, index) => (
                      <List.Item>
                        <Text>
                          {index + 1}. {recommendation}
                        </Text>
                      </List.Item>
                    )}
                  />
                </Card>
              </Col>
            </Row>
          )}
        </>
      )}
    </div>
  );
};

export default ComparativeResults;
