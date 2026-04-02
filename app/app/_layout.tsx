import { Stack } from 'expo-router';
import { UploadProvider } from '../context/UploadContext';
import { AuthProvider } from '../context/AuthContext';

export default function RootLayout() {
  return (
    <AuthProvider>
      <UploadProvider>
        <Stack />
      </UploadProvider>
    </AuthProvider>
  );
}

