import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Layout } from "antd";
import { AnalysisProvider } from "./context/AnalysisContext";
import Sidebar from "./components/Sidebar";
import AnalysisForm from "./components/AnalysisForm";
import FilterResults from "./components/results/FilterResults";
import FrameworkerResults from "./components/results/FrameworkerResults";
import ClassifierResults from "./components/results/ClassifierResults";
import WizardResults from "./components/results/WizardResults";
import ComparativeResults from "./components/results/ComparativeResults";

const { Content, Sider } = Layout;

function App() {
  return (
    <Router>
      <AnalysisProvider>
        <Layout style={{ minHeight: "100vh" }}>
          <Sider width={300} theme="light">
            <Sidebar />
          </Sider>
          <Layout>
            <Content
              style={{ padding: "24px", minHeight: 280, width: "800px" }}
            >
              <Routes>
                <Route path="/" element={<AnalysisForm />} />
                <Route path="/results/filter/:id" element={<FilterResults />} />
                <Route
                  path="/results/frameworker/:id"
                  element={<FrameworkerResults />}
                />
                <Route
                  path="/results/classifier/:id"
                  element={<ClassifierResults />}
                />
                <Route path="/results/wizard/:id" element={<WizardResults />} />
                <Route
                  path="/results/comparative/:id"
                  element={<ComparativeResults />}
                />
              </Routes>
            </Content>
          </Layout>
        </Layout>
      </AnalysisProvider>
    </Router>
  );
}

export default App;
