import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { DashboardPage } from './pages/DashboardPage'
import { SourcesPage } from './pages/SourcesPage'
import { ModelsPage } from './pages/ModelsPage'
import { PipelinesPage } from './pages/PipelinesPage'
import { TasksPage } from './pages/TasksPage'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/sources" element={<SourcesPage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/pipelines" element={<PipelinesPage />} />
        <Route path="/tasks" element={<TasksPage />} />
      </Routes>
    </Layout>
  )
}
