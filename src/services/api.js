import axios from "axios";


export const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";


const api = axios.create({

  baseURL: API_URL,

  timeout: 120000

});


function createImageFormData(
  file
) {

  const formData =
    new FormData();


  formData.append(
    "file",
    file
  );


  return formData;
}


export async function getHealth() {

  const response =
    await api.get(
      "/health"
    );


  return response.data;
}


export async function getClasses() {

  const response =
    await api.get(
      "/classes"
    );


  return response.data;
}


export async function predictDefects(
  file,
  confidence = 0.25
) {

  const response =
    await api.post(
      "/predict",
      createImageFormData(
        file
      ),
      {
        params: {
          confidence
        }
      }
    );


  return response.data;
}


export async function predictAnnotatedImage(
  file,
  confidence = 0.25
) {

  const response =
    await api.post(
      "/predict/image",
      createImageFormData(
        file
      ),
      {
        params: {
          confidence
        },

        responseType:
          "blob"
      }
    );


  return response.data;
}


export function getApiErrorMessage(
  error
) {

  const detail =
    error?.response?.data
      ?.detail;


  if (
    typeof detail ===
    "string"
  ) {

    return detail;
  }


  if (
    error?.code ===
    "ECONNABORTED"
  ) {

    return (
      "The backend request timed out."
    );
  }


  if (
    !error?.response
  ) {

    return (
      "Unable to connect to the AI backend. "
      + "Make sure FastAPI is running."
    );
  }


  if (
    error.response.status ===
    503
  ) {

    return (
      "The AI model is not available on "
      + "the backend."
    );
  }


  return (
    `Backend request failed with status `
    + `${error.response.status}.`
  );
}