terraform {
  required_providers {
    kind = {
      source  = "tehcyx/kind"
      version = "~> 0.0.17"
    }
    flux = {
      source  = "fluxcd/flux"
      version = "~> 1.2"
    }
    github = {
      source  = "integrations/github"
      version = "~> 6.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

provider "kind" {}

provider "flux" {}

provider "github" {
  owner = var.github_owner
  token = var.github_token
}

provider "kubernetes" {
  host                   = module.kind_cluster.endpoint
  cluster_ca_certificate = module.kind_cluster.cluster_ca_certificate
  client_certificate     = module.kind_cluster.client_certificate
  client_key             = module.kind_cluster.client_key
}
