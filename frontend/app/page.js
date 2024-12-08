"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";

export default function Chatbot() {
  const [currentTime, setCurrentTime] = useState('Loading...');
  const [messages, setMessages] = useState([]); // Chat history
  const [input, setInput] = useState(""); // User query
  const [loading, setLoading] = useState(false); // Loading state
  const [contextURL, setContextURL] = useState(""); // Context URL
  const [mode, setMode] = useState("pretrain"); // Mode (RAG or Pretrained)
  const chatContainerRef = useRef(null); // Reference for scrolling
  const [isClient, setIsClient] = useState(false);

  // Ensure the component runs only on the client
  useEffect(() => {
    setIsClient(true);
    setCurrentTime(new Date().toLocaleString());
  }, []);

  // Automatically scroll to the bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() && !contextURL.trim()) return; // Do nothing if input is empty

    setLoading(true);

    try {
      if (contextURL.trim()) {
        const endpoint = mode === "pretrain" ? "/pretrain" : "/add_to_kg";
        const newMessages = [
          ...messages,
          {
            sender: "user",
            text: `Context: ${contextURL}\nQuery: ${input.trim()}`,
          },
        ];
        setMessages(newMessages); // Add user input to chat history
        setInput(""); // Clear input
        setContextURL("");

        const typingMessage = { sender: "bot", text: "Training..." };
        setMessages((prevMessages) => [...prevMessages, typingMessage]);

        const response = await axios.post(`http://localhost:5000${endpoint}`, {
          url: contextURL,
        });

        const botMessage = "Model Updated!";
        setMessages((prevMessages) =>
          prevMessages.map((msg, index) =>
            index === prevMessages.length - 1
              ? { sender: "bot", text: botMessage }
              : msg
          )
        );
      }

      if (input.trim()) {
        const newMessages = [
          ...messages,
          {
            sender: "user",
            text: `${input.trim()}`,
          },
        ];
        setMessages(newMessages); // Add user input to chat history
        setInput(""); // Clear input
        setContextURL("");
        const typingMessage = { sender: "bot", text: "Typing..." };
        setMessages((prevMessages) => [...prevMessages, typingMessage]);

        const response = await axios.post("http://localhost:5000/chat", {
          query: input,
        });

        const botMessage = response.data.response;
        setMessages((prevMessages) =>
          prevMessages.map((msg, index) =>
            index === prevMessages.length - 1
              ? { sender: "bot", text: botMessage }
              : msg
          )
        );
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "bot", text: "Error: Unable to fetch response." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  if (!isClient) {
    // Render a placeholder during SSR
    return <div>Loading...</div>;
  }

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="bg-blue-600 text-white p-4 text-center font-bold">
        AI Chatbot
      </header>
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`${
                message.sender === "user" ? "bg-blue-500" : "bg-gray-300"
              } text-white rounded-lg p-3 max-w-xs`}
            >
              {message.text}
            </div>
          </div>
        ))}
      </div>
      <div className="p-4 bg-white space-y-4 border-t">
        <div className="flex items-center space-x-2">
          <label className="font-bold">Context URL:</label>
          <input
            type="text"
            className="flex-1 border rounded-lg p-2 focus:outline-none"
            placeholder="Enter context URL..."
            value={contextURL}
            onChange={(e) => setContextURL(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
        </div>
        <div className="flex items-center space-x-4">
          <label className="font-bold">Mode:</label>
          <div className="flex items-center space-x-2">
            <input
              type="radio"
              id="pretrain"
              name="mode"
              value="pretrain"
              checked={mode === "pretrain"}
              onChange={(e) => setMode(e.target.value)}
            />
            <label htmlFor="pretrain">Pretrained</label>
          </div>
          <div className="flex items-center space-x-2">
            <input
              type="radio"
              id="rag"
              name="mode"
              value="rag"
              checked={mode === "rag"}
              onChange={(e) => setMode(e.target.value)}
            />
            <label htmlFor="rag">RAG</label>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <input
            type="text"
            className="flex-1 border rounded-lg p-2 focus:outline-none"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
          />
          <button
            onClick={handleSend}
            className="bg-blue-600 text-white py-2 px-4 rounded-lg"
            disabled={loading}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}