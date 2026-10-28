import React from 'react';
import Homepage from './pages/Homepage';

function App() {
  return (
    <div className="App">
      {/* In the future, you can wrap <Homepage /> with a Router 
          if you add pages like /gallery or /dashboard. 
      */}
      <Homepage />
    </div>
  );
}

export default App;