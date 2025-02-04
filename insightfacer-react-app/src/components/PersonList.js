import React, { useState, useEffect } from "react";

function PersonList() {
  const [persons, setPersons] = useState([]);

  useEffect(() => {
    const fetchPersons = async () => {
      try {
        const response = await fetch("/list_persons");
        const data = await response.json();
        setPersons(data.persons);
      } catch (error) {
        console.error("Error fetching person list:", error);
      }
    };
    fetchPersons();
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h1>已註冊人員</h1>
      {persons.length > 0 ? (
        <ul>
          {persons.map((p, index) => (
            <li key={index}>{p}</li>
          ))}
        </ul>
      ) : (
        <p>尚無註冊人員</p>
      )}
    </div>
  );
}

export default PersonList;
