import React from "react";
import { Menu, Typography, Divider } from "antd";
import { FileOutlined, HistoryOutlined } from "@ant-design/icons";
import { useAnalysis } from "../context/AnalysisContext";
import { formatFileSize, formatDate } from "../utils/format";

const { Title } = Typography;

const Sidebar: React.FC = () => {
  const { state } = useAnalysis();

  return (
    <div style={{ padding: "16px" }}>
      <Title level={4}>Available Files</Title>
      <Menu mode="inline">
        {state.files.map((file) => (
          <Menu.Item key={file.name} icon={<FileOutlined />}>
            <div>
              <div>{file.name}</div>
              <div style={{ fontSize: "12px", color: "#999" }}>
                {formatFileSize(file.size)} • {formatDate(file.modified)}
              </div>
            </div>
          </Menu.Item>
        ))}
      </Menu>

      <Divider />

      <Title level={4}>Recent Analyses</Title>
      <Menu mode="inline">
        {/* We'll implement this later when we add history tracking */}
        <Menu.Item key="placeholder" icon={<HistoryOutlined />} disabled>
          No recent analyses
        </Menu.Item>
      </Menu>
    </div>
  );
};

export default Sidebar;
