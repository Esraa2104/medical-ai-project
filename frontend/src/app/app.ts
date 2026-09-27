import { Component, signal, computed } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';

interface PredictionResult {
  prediction: string;
  probability_normal: number;
  probability_pneumonia: number;
  model_name: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [HttpClientModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  selectedFile: File | null = null;
  fileName = signal<string | null>(null);
  imagePreview = signal<string | ArrayBuffer | null>(null);
  result = signal<PredictionResult | null>(null);
  error = signal<string | null>(null);
  loading = signal(false);
  analyzedAt = signal<string | null>(null);

  normalPct = computed(() => {
    const r = this.result();
    return r ? Math.round(r.probability_normal * 1000) / 10 : 0;
  });
  pneumoniaPct = computed(() => {
    const r = this.result();
    return r ? Math.round(r.probability_pneumonia * 1000) / 10 : 0;
  });
  isPneumonia = computed(() => this.result()?.prediction === 'PNEUMONIA');

  private apiUrl = 'http://127.0.0.1:8000/predict';

  constructor(private http: HttpClient) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.fileName.set(this.selectedFile.name);
      this.result.set(null);
      this.error.set(null);

      const reader = new FileReader();
      reader.onload = () => this.imagePreview.set(reader.result);
      reader.readAsDataURL(this.selectedFile);
    }
  }

  analyzeImage(): void {
    if (!this.selectedFile) return;

    this.loading.set(true);
    this.error.set(null);
    this.result.set(null);

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.http.post<PredictionResult>(this.apiUrl, formData).subscribe({
      next: (res) => {
        this.result.set(res);
        this.loading.set(false);
        this.analyzedAt.set(new Date().toLocaleString('fr-FR'));
      },
      error: (err) => {
        this.error.set('Le service de lecture est injoignable. Vérifiez que l\'API tourne sur le port 8000.');
        this.loading.set(false);
        console.error(err);
      }
    });
  }

  reset(): void {
    this.selectedFile = null;
    this.fileName.set(null);
    this.imagePreview.set(null);
    this.result.set(null);
    this.error.set(null);
  }
}