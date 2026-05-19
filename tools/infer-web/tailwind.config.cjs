/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          base: '#0b0f14',
          surface: '#111720',
          overlay: '#19212e',
          elevated: '#202b3a',
        },
        ink: {
          primary: '#dde5f0',
          secondary: '#8393a8',
          muted: '#465669',
        },
        accent: {
          DEFAULT: '#22d3ee',
          dim: '#0891b2',
          glow: 'rgba(34,211,238,0.12)',
        },
        success: '#34d399',
        warning: '#fbbf24',
        danger: '#f87171',
        border: '#1e2d3d',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        sm: '2px',
        DEFAULT: '3px',
        lg: '5px',
        xl: '8px',
      },
      animation: {
        'pulse-slow': 'pulse 2.4s cubic-bezier(0.4,0,0.6,1) infinite',
      },
    },
  },
  plugins: [],
}
