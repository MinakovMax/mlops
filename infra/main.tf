# ─────────────────────────────────────────────────────────────
# Создание виртуальной сети (VPC)
# ─────────────────────────────────────────────────────────────
resource "yandex_vpc_network" "default" {}

# Создание подсети в зоне ru-central1-a
resource "yandex_vpc_subnet" "default" {
  zone           = "ru-central1-a"
  network_id     = yandex_vpc_network.default.id
  v4_cidr_blocks = ["10.2.0.0/24"]
}

# ─────────────────────────────────────────────────────────────
# PostgreSQL-кластер для MLFlow и хранения обучающих данных
# ─────────────────────────────────────────────────────────────
resource "yandex_mdb_postgresql_cluster" "pg" {
  name        = "mlflow-db"
  environment = "PRODUCTION"
  network_id  = yandex_vpc_network.default.id

  config {
    version = 14
    resources {
      resource_preset_id = "s2.micro"       # дешёвый вариант для старта
      disk_size          = 20               # размер в ГБ
      disk_type_id       = "network-ssd"
    }
  }

  host {
    zone      = "ru-central1-a"
    subnet_id = yandex_vpc_subnet.default.id
    name      = "pg-host"
  }
}

# Создание базы данных внутри кластера
resource "yandex_mdb_postgresql_database" "mlflow" {
  cluster_id = yandex_mdb_postgresql_cluster.pg.id
  name       = "mlflow"
  owner      = var.db_user
}

# Создание пользователя с паролем (берётся из переменной окружения)
resource "yandex_mdb_postgresql_user" "mlflow" {
  cluster_id = yandex_mdb_postgresql_cluster.pg.id
  name       = var.db_user
  password   = var.db_password
}

# ─────────────────────────────────────────────────────────────
# S3-хранилище для моделей, артефактов и логов MLFlow
# ─────────────────────────────────────────────────────────────
resource "random_id" "bucket_id" {
  byte_length = 4
}

resource "yandex_storage_bucket" "models_bucket" {
  bucket   = "ml-models-${random_id.bucket_id.hex}" # уникальное имя
  max_size = 10737418240  # 10 ГБ
  acl      = "private"
}

# ─────────────────────────────────────────────────────────────
# Kubernetes-кластер для инференса и MLFlow
# ─────────────────────────────────────────────────────────────
resource "yandex_kubernetes_cluster" "mlops_cluster" {
  name       = "mlops-cluster"
  network_id = yandex_vpc_network.default.id

  master {
    version = "1.30"
    zonal {
      zone      = "ru-central1-a"
      subnet_id = yandex_vpc_subnet.default.id
    }

    public_ip = true
  }

  service_account_id      = var.k8s_sa_id
  node_service_account_id = var.k8s_sa_id

  release_channel         = "RAPID"
  network_policy_provider = "CALICO"
}

# Группа рабочих узлов для запуска подов
resource "yandex_kubernetes_node_group" "default" {
  cluster_id = yandex_kubernetes_cluster.mlops_cluster.id
  name       = "default-pool"
  version = "1.30"

  instance_template {
    platform_id = "standard-v2"
    resources {
      memory = 4
      cores  = 2
    }

    boot_disk {
      type = "network-ssd"
      size = 50
    }

    network_interface {
      subnet_ids = [yandex_vpc_subnet.default.id]
      nat        = true
    }

    scheduling_policy {
      preemptible = true # более дешёвые, но нестабильные ВМ
    }
  }

  scale_policy {
    fixed_scale {
      size = 2 # количество рабочих узлов
    }
  }

  allocation_policy {
    location {
      zone = "ru-central1-a"
    }
  }
}