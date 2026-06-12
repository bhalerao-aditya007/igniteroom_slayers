import { useState } from 'react'
import { Stethoscope, Pill, User, Activity, ArrowLeft, ExternalLink, Shield, HeartPulse } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import './App.css'

type View = 'main' | 'doctor'

interface MainButtonProps {
  title: string
  icon: React.ReactNode
  onClick: () => void
  color: string
  description: string
}

const MainButton = ({ title, icon, onClick, color, description }: MainButtonProps) => (
  <Card 
    className={`group cursor-pointer transition-all duration-500 hover:scale-105 hover:shadow-2xl border-0 overflow-hidden ${color}`}
    onClick={onClick}
  >
    <CardContent className="p-8 flex flex-col items-center text-center h-full min-h-[280px] justify-center relative">
      <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      <div className="relative z-10">
        <div className="mb-6 transform group-hover:scale-110 transition-transform duration-500">
          {icon}
        </div>
        <h3 className="text-2xl font-bold text-white mb-3 tracking-wide">{title}</h3>
        <p className="text-white/80 text-sm leading-relaxed">{description}</p>
      </div>
    </CardContent>
  </Card>
)

interface LinkButtonProps {
  title: string
  url: string
  icon: React.ReactNode
  color: string
  description: string
}

const LinkButton = ({ title, url, icon, color, description }: LinkButtonProps) => (
  <a 
    href={url}
    target="_blank"
    rel="noopener noreferrer"
    className="block"
  >
    <Card 
      className={`group cursor-pointer transition-all duration-500 hover:scale-105 hover:shadow-2xl border-0 overflow-hidden ${color}`}
    >
      <CardContent className="p-8 flex flex-col items-center text-center h-full min-h-[240px] justify-center relative">
        <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        <div className="relative z-10">
          <div className="mb-6 transform group-hover:scale-110 transition-transform duration-500">
            {icon}
          </div>
          <h3 className="text-2xl font-bold text-white mb-3 tracking-wide">{title}</h3>
          <p className="text-white/80 text-sm leading-relaxed mb-4">{description}</p>
          <ExternalLink className="w-5 h-5 text-white/60 mx-auto group-hover:text-white transition-colors" />
        </div>
      </CardContent>
    </Card>
  </a>
)

function App() {
  const [currentView, setCurrentView] = useState<View>('main')

  const handleDoctorClick = () => {
    setCurrentView('doctor')
  }

  const handleBackToMain = () => {
    setCurrentView('main')
  }

  return (
    <div className="min-h-screen w-full relative overflow-hidden">
      {/* Background Image */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: 'url(/hiv-background.jpg)' }}
      />
      
      {/* Dark Overlay */}
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />
      
      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Header */}
        <header className="w-full py-6 px-8">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-red-400" />
              <h1 className="text-2xl font-bold text-white tracking-wider">
                HIV Healthcare Portal
              </h1>
            </div>
            {currentView === 'doctor' && (
              <Button
                variant="ghost"
                onClick={handleBackToMain}
                className="text-white hover:bg-white/20 flex items-center gap-2"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Main
              </Button>
            )}
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 flex items-center justify-center px-4 sm:px-6 lg:px-8 py-12">
          <div className="w-full max-w-6xl">
            {currentView === 'main' ? (
              <div className="space-y-12">
                {/* Welcome Section */}
                <div className="text-center space-y-4">
                  <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-red-500/20 border border-red-400/30">
                    <HeartPulse className="w-4 h-4 text-red-400" />
                    <span className="text-red-200 text-sm font-medium">Comprehensive HIV Care Platform</span>
                  </div>
                  <h2 className="text-4xl md:text-5xl font-bold text-white">
                    Select Your Role
                  </h2>
                  <p className="text-white/70 text-lg max-w-2xl mx-auto">
                    Access specialized tools and resources tailored for healthcare professionals and patients
                  </p>
                </div>

                {/* Buttons Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <MainButton
                    title="Doctor"
                    icon={<Stethoscope className="w-16 h-16 text-white" />}
                    onClick={handleDoctorClick}
                    color="bg-gradient-to-br from-blue-600 to-blue-800"
                    description="Access clinical support tools and risk assessment resources"
                  />
                  <a 
                    href="https://hiv-drug-simulation.vercel.app/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block"
                  >
                    <MainButton
                      title="Pharmacist"
                      icon={<Pill className="w-16 h-16 text-white" />}
                      onClick={() => {}}
                      color="bg-gradient-to-br from-emerald-600 to-emerald-800"
                      description="Drug interaction simulation and medication management"
                    />
                  </a>
                  <a 
                    href="https://remix-of-patient-health-hub.vercel.app/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block"
                  >
                    <MainButton
                      title="Patient"
                      icon={<User className="w-16 h-16 text-white" />}
                      onClick={() => {}}
                      color="bg-gradient-to-br from-purple-600 to-purple-800"
                      description="Personal health hub and patient care resources"
                    />
                  </a>
                  <a 
                    href="https://heart-risk.lovable.app"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block"
                  >
                    <MainButton
                      title="Other Diseases"
                      icon={<Activity className="w-16 h-16 text-white" />}
                      onClick={() => {}}
                      color="bg-gradient-to-br from-amber-600 to-amber-800"
                      description="Resources for other infectious diseases and conditions"
                    />
                  </a>
                </div>
              </div>
            ) : (
              <div className="space-y-12">
                {/* Doctor Submenu */}
                <div className="text-center space-y-4">
                  <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/20 border border-blue-400/30">
                    <Stethoscope className="w-4 h-4 text-blue-400" />
                    <span className="text-blue-200 text-sm font-medium">Doctor Portal</span>
                  </div>
                  <h2 className="text-4xl md:text-5xl font-bold text-white">
                    Clinical Resources
                  </h2>
                  <p className="text-white/70 text-lg max-w-2xl mx-auto">
                    Choose from specialized clinical tools and assessment applications
                  </p>
                </div>

                {/* Doctor Buttons Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-3xl mx-auto">
                  <LinkButton
                    title="Clinical Support"
                    url="https://clean-site-delight.vercel.app/"
                    icon={<Shield className="w-16 h-16 text-white" />}
                    color="bg-gradient-to-br from-cyan-600 to-cyan-800"
                    description="Access comprehensive clinical support and decision-making tools"
                  />
                  <LinkButton
                    title="Risk Assessment"
                    url="https://clinical-progression-risk-assessmen.vercel.app/"
                    icon={<Activity className="w-16 h-16 text-white" />}
                    color="bg-gradient-to-br from-rose-600 to-rose-800"
                    description="Clinical progression risk assessment and monitoring tools"
                  />
                </div>
              </div>
            )}
          </div>
        </main>

        {/* Footer */}
        <footer className="w-full py-6 px-8">
          <div className="max-w-7xl mx-auto text-center">
            <p className="text-white/50 text-sm">
              HIV Healthcare Portal — Empowering Care Through Technology
            </p>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default App
