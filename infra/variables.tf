variable "db_user" {
  description = "PostgreSQL user"
  type        = string
  default     = "mlflow"
}

variable "db_password" {
  description = "PostgreSQL password"
  type        = string
  sensitive   = true
}

variable "k8s_sa_id" {
  description = "ID сервисного аккаунта для Kubernetes"
  type        = string
}