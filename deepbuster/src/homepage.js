import React, { useEffect, useState } from "react";

function Homepage() {
  const [message, setMessage] = useState("");

  useEffect(() => {
    fetch("/api/message")
      .then((response) => response.json())
      .then((data) => setMessage(data.message))
      .catch((error) => console.error("Error fetching message:", error));
  }, []);

  return (
    <div>
      <h1>Welcome to DeepBuster!</h1>
      <p>{message}</p>
    </div>
  );
}

export default Homepage;