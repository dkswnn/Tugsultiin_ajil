import { useEffect, useState } from "react";

interface AttendanceRecord {
  _id: string;
  studentId: string;
  filename: string;
  score: number;
  timestamp: string;
  isPredictionCorrect?: boolean;
  notes?: string;
}

interface StudentInfo {
  _id: string;
  studentId: string;
  name: string;
  totalAttendanceCount: number;
  lastAttendanceDate?: string;
}

const API_BASE = "http://localhost:3000";

export default function App() {
  const [attendanceRecords, setAttendanceRecords] = useState<
    AttendanceRecord[]
  >([]);
  const [students, setStudents] = useState<StudentInfo[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>(
    new Date().toISOString().split("T")[0],
  );
  const [isAdmin, setIsAdmin] = useState(false);
  const [loading, setLoading] = useState(false);

  // Fetch attendance records for selected date
  const fetchAttendanceByDate = async () => {
    setLoading(true);
    try {
      const res = await fetch(
        `${API_BASE}/attendance/date?date=${selectedDate}`,
      );
      const data = await res.json();
      setAttendanceRecords(data);
    } catch (error) {
      console.error("Error fetching attendance:", error);
    }
    setLoading(false);
  };

  // Fetch all students
  const fetchStudents = async () => {
    try {
      const res = await fetch(`${API_BASE}/students`);
      const data = await res.json();
      setStudents(data);
    } catch (error) {
      console.error("Error fetching students:", error);
    }
  };

  // Mark prediction as correct/incorrect (admin only)
  const updatePredictionCorrectness = async (
    recordId: string,
    isCorrect: boolean,
    notes: string = "",
  ) => {
    try {
      const res = await fetch(
        `${API_BASE}/attendance/${recordId}/correctness`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ isPredictionCorrect: isCorrect, notes }),
        },
      );
      const updated = await res.json();
      setAttendanceRecords((records) =>
        records.map((r) => (r._id === recordId ? updated : r)),
      );
    } catch (error) {
      console.error("Error updating correctness:", error);
    }
  };

  // Load initial data
  useEffect(() => {
    fetchStudents();
    fetchAttendanceByDate();
  }, [selectedDate]);

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>Attendance Recognition Dashboard</h1>

      {/* Mode Toggle */}
      <div style={{ marginBottom: "20px" }}>
        <button
          onClick={() => setIsAdmin(!isAdmin)}
          style={{ padding: "10px 20px" }}
        >
          {isAdmin ? "Switch to Teacher View" : "Switch to Admin View"}
        </button>
      </div>

      {/* Teacher View */}
      {!isAdmin && (
        <div>
          <h2>Attendance Records</h2>
          <div style={{ marginBottom: "20px" }}>
            <label>
              Select Date:{" "}
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
              />
            </label>
            <button
              onClick={fetchAttendanceByDate}
              style={{ marginLeft: "10px" }}
            >
              Refresh
            </button>
          </div>

          {loading ? (
            <p>Loading...</p>
          ) : attendanceRecords.length === 0 ? (
            <p>No records for selected date</p>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "2px solid black" }}>
                  <th style={{ padding: "10px", textAlign: "left" }}>Time</th>
                  <th style={{ padding: "10px", textAlign: "left" }}>
                    Student ID
                  </th>
                  <th style={{ padding: "10px", textAlign: "left" }}>
                    Confidence
                  </th>
                  <th style={{ padding: "10px", textAlign: "left" }}>Image</th>
                </tr>
              </thead>
              <tbody>
                {attendanceRecords.map((record) => (
                  <tr
                    key={record._id}
                    style={{ borderBottom: "1px solid #ddd" }}
                  >
                    <td style={{ padding: "10px" }}>
                      {new Date(record.timestamp).toLocaleTimeString()}
                    </td>
                    <td style={{ padding: "10px" }}>{record.studentId}</td>
                    <td style={{ padding: "10px" }}>
                      {(record.score * 100).toFixed(2)}%
                    </td>
                    <td style={{ padding: "10px" }}>{record.filename}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h2>Students ({students.length})</h2>
          <ul>
            {students.map((student) => (
              <li key={student._id}>
                {student.name} ({student.studentId}) -{" "}
                {student.totalAttendanceCount} records
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Admin View */}
      {isAdmin && (
        <div>
          <h2>Admin Panel - Review Predictions</h2>
          <p style={{ color: "#666" }}>
            Review and mark predictions as True Positive (TP), False Positive
            (FP), etc.
          </p>

          {attendanceRecords.length === 0 ? (
            <p>No records to review</p>
          ) : (
            <div>
              {attendanceRecords.map((record) => (
                <div
                  key={record._id}
                  style={{
                    border: "1px solid #ccc",
                    padding: "15px",
                    marginBottom: "15px",
                  }}
                >
                  <p>
                    <strong>{record.studentId}</strong> - {record.filename} -
                    Confidence: {(record.score * 100).toFixed(2)}%
                  </p>
                  <p>
                    Timestamp: {new Date(record.timestamp).toLocaleString()}
                  </p>
                  <div style={{ marginBottom: "10px" }}>
                    <label>Status: </label>
                    {record.isPredictionCorrect === true && (
                      <span style={{ color: "green", fontWeight: "bold" }}>
                        ✓ Correct (TP)
                      </span>
                    )}
                    {record.isPredictionCorrect === false && (
                      <span style={{ color: "red", fontWeight: "bold" }}>
                        ✗ Incorrect (FP/FN)
                      </span>
                    )}
                    {record.isPredictionCorrect === undefined && (
                      <span style={{ color: "gray" }}>- Not Reviewed</span>
                    )}
                  </div>
                  <div style={{ marginBottom: "10px" }}>
                    <button
                      onClick={() =>
                        updatePredictionCorrectness(record._id, true)
                      }
                      style={{
                        padding: "8px 12px",
                        marginRight: "10px",
                        backgroundColor:
                          record.isPredictionCorrect === true
                            ? "#4CAF50"
                            : "#f0f0f0",
                        color:
                          record.isPredictionCorrect === true
                            ? "white"
                            : "black",
                        border: "1px solid #ccc",
                        cursor: "pointer",
                      }}
                    >
                      Mark Correct ✓
                    </button>
                    <button
                      onClick={() =>
                        updatePredictionCorrectness(record._id, false)
                      }
                      style={{
                        padding: "8px 12px",
                        backgroundColor:
                          record.isPredictionCorrect === false
                            ? "#f44336"
                            : "#f0f0f0",
                        color:
                          record.isPredictionCorrect === false
                            ? "white"
                            : "black",
                        border: "1px solid #ccc",
                        cursor: "pointer",
                      }}
                    >
                      Mark Incorrect ✗
                    </button>
                  </div>
                  {record.notes && <p>Notes: {record.notes}</p>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
