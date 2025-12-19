import torch
import torch.nn as nn
import torch.nn.functional as F
from module.ldm.util import checkpoint

def cos_sim(A, B, dim= -1, eps= 1e-4):

    # ℓ2-norm
    norm_A = F.normalize(A, p=2, dim=dim, eps=eps)
    norm_B = F.normalize(B, p=2, dim=dim, eps=eps)

    # sim = torch.matmul(norm_A, norm_B.transpose(-1, -2))
    return norm_A @ norm_B.transpose(-1, -2)



class GeometryPreservingLoss(nn.Module):
    """
    - eeg_latent : (B, N, D)  — EEG encoder 출력
    - text_latent : (B, N, D)  — Text encoder 출력
    - text_inputs: (B, S) - Text 입력
    - eeg_inputs: (B, S, H) - EEG 입력

    """
    def __init__(self):
        super().__init__()

    def forward(self, eeg_latent, txt_emb, mask=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (eeg_latent, txt_emb, mask), self.parameters(), True
        )
    
    def _forward(self, eeg_latent, txt_emb, mask=None):

        
        # ----- 1. similarity matrix W (cosine) -----
        W = cos_sim(txt_emb, txt_emb) # (B, N, N) 
        
        # ----- 2. 그래프 라플라시안 L = D − W -----
        D = torch.diag_embed(W.sum(-1)) # degree matrix
        
        L = D - W  # (B, N, N)
        # L = F.normalize(L, p=2, dim=-1, eps=eps)

        Z = eeg_latent # (B, N, D)
        # Z = F.normalize(eeg_latent, p=2, dim=-1, eps=eps)

        # ----- 3. Geometry-preserving loss -----
        # ZT_L = torch.matmul(Z.transpose(-1,-2), L)    
        # ZT_L_Z = torch.matmul(ZT_L, Z) # (B, D, D)
        ZT_L = Z.transpose(-1,-2) @ L
        ZT_L_Z = ZT_L @ Z
        loss = torch.diagonal(ZT_L_Z, dim1=-2, dim2=-1).sum(-1, keepdim=True)

        if mask is not None:
            mask_flat = mask.reshape(-1)
            loss = (loss * mask_flat).sum() / mask_flat.sum()

        # loss = torch.log1p(loss)
        return loss
    
class ContrastiveLatentLoss(nn.Module):
    def __init__(self, gamma=0.07):
        """
        Contrastive loss for aligning EEG Condition with Text Latent in Latent Diffusion Model (LDM).
        """
        super(ContrastiveLatentLoss, self).__init__()
        self.gamma = gamma
        
        hidden_dim = 512 
        projection_dim = 256
        # self.text_projection = nn.Sequential(
        #     nn.Linear(hidden_dim, projection_dim),
        #     nn.ReLU(),
        #     nn.Linear(projection_dim, hidden_dim)
        # )
        # self.eeg_projection = nn.Sequential(
        #     nn.Linear(hidden_dim, projection_dim),
        #     nn.ReLU(),
        #     nn.Linear(projection_dim, hidden_dim)
        # )

        self.text_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.eeg_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        # self.conv1x1 = nn.Conv1d(56, 1, 1)
        # self.seq_len = seq_len
        # self.batch_size = batch_size


    def forward(self, text_latent, eeg_condition, mask=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (text_latent, eeg_condition), self.parameters(), True
        )
    
    def _forward(self, text_latent, eeg_condition, mask=None):
        """
        Args:
            text_latent (torch.Tensor): Latent representation of text (batch_size, latent_dim)
            eeg_condition (torch.Tensor): EEG-based Condition Vector (batch_size, latent_dim)
        
        Returns:
            torch.Tensor: Contrastive loss value
        """
        # batch_size, seq_len, hidden_dim = text_latent.shape

        # # (2) 각 문장 내 단어 간 similarity 계산
        # similarity = cos_sim(text_latent, eeg_condition) / self.gamma

        # # (3) 정답 labels 생성 (각 단어가 동일 위치 단어와 매칭되도록)
        # labels = torch.arange(seq_len).unsqueeze(0).repeat(batch_size,1).to(eeg_condition.device)
        # # labels shape: [batch_size, seq_len] (정답은 항상 [0,1,...,seq_len-1])

        # # (4) loss 계산을 위해 shape 변환
        # similarity_flat = similarity.reshape(batch_size * seq_len, seq_len)
        # labels_flat = labels.reshape(batch_size * seq_len)

        # # (5) Contrastive loss 계산 (단어 수준 Contrastive loss)
        # loss = F.cross_entropy(similarity_flat, labels_flat)
        # # loss = F.cross_entropy(similarity_flat, labels_flat, reduction='mean', ignore_index=-100)
        
        # if mask is not None:
        #     mask_flat = mask.reshape(-1)
        #     loss = (loss * mask_flat).sum() / mask_flat.sum()

################################################################################################################

        batch_size, _, _ = text_latent.shape
        text_sent_emb = text_latent.mean(dim=1)  # (batch_size, hidden_dim)
        eeg_sent_emb = eeg_condition.mean(dim=1) # (batch_size, hidden_dim)
        # # text_sent_emb = self.conv1x1(text_latent).squeeze(1)  # (batch_size, hidden_dim)
        # # eeg_sent_emb = self.conv1x1(eeg_condition).squeeze(1) # (batch_size, hidden_dim)
        text_embedding = self.text_projection(text_sent_emb)
        eeg_embedding = self.eeg_projection(eeg_sent_emb)


        # (2) 각 문장 내 단어 간 similarity 계산
        # similarity = cos_sim(text_sent_emb, eeg_sent_emb) / self.gamma
        similarity = cos_sim(text_embedding, eeg_embedding) / self.gamma

        # (3) 정답 labels 생성 (각 단어가 동일 위치 단어와 매칭되도록)
        labels = torch.arange(batch_size).to(eeg_sent_emb.device)
        # labels shape: [batch_size, seq_len] (정답은 항상 [0,1,...,seq_len-1])

        # (5) Contrastive loss 계산 (단어 수준 Contrastive loss)
        loss = F.cross_entropy(similarity, labels)

        return loss