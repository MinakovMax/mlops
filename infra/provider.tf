terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = "~> 0.99"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7"
    }
  }
}

provider "yandex" {
  service_account_key_file = "${path.module}/key.json"
  cloud_id  = "b1geqc3j25h2f34cha4p"
  folder_id = "b1gag8ikm3e5t7t73a71"
  zone      = "ru-central1-a"
}