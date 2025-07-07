import { useState, useEffect } from 'react';
import Head from 'next/head';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [pdfList, setPdfList] = useState<string[]>([]);

  useEffect(() => {
    fetch('/api/files')
      .then(res => res.json())
      .then(setPdfList)
      .catch(() => setPdfList([]));
  }, []);

  const sendMessage = async () => {
    if (!input) return;
    setMessages([...messages, { role: 'user', content: input }]);
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: input })
    });
    const data = await res.json();
    setMessages(m => [...m, { role: 'assistant', content: data.reply }]);
    setInput('');
  };

  return (
    <div>
      <Head>
        <title>Sales Genius</title>
      </Head>
      <h1>Sales Genius</h1>
      <div>
        <input value={input} onChange={e => setInput(e.target.value)} />
        <button onClick={sendMessage}>Send</button>
      </div>
      <div>
        {messages.map((m, i) => (
          <p key={i}><b>{m.role}:</b> {m.content}</p>
        ))}
      </div>
      <h2>Available PDFs</h2>
      <ul>
        {pdfList.map(f => <li key={f}>{f}</li>)}
      </ul>
    </div>
  );
}
