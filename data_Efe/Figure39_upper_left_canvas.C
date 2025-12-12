#ifdef __CLING__
#pragma cling optimize(0)
#endif
void Figure39_upper_left_canvas()
{
//=========Macro generated from canvas: c/
//=========  (Thu Nov 20 14:00:47 2025) by ROOT version 6.30/07
   TCanvas *c = new TCanvas("c", "",0,0,1280,1024);
   c->SetHighLightColor(2);
   c->Range(-15,-0.1875,135,1.6875);
   c->SetFillColor(0);
   c->SetBorderMode(0);
   c->SetBorderSize(2);
   c->SetFrameBorderMode(0);
   c->SetFrameBorderMode(0);
   
   Double_t divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_fx3001[30] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 36,
   40, 44, 48, 52.5, 57.5, 62.5, 67.5, 75, 85, 95, 110, 135, 175 };
   Double_t divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_fy3001[30] = { 0.895924, 0.8863045, 0.8812615, 0.8739463, 0.8681265, 0.860023, 0.8527678, 0.8445349, 0.8345187, 0.8316689, 0.8282407, 0.8179812, 0.8055721, 0.8021212, 0.7930767, 0.7814584, 0.7717184,
   0.769256, 0.7624505, 0.7552796, 0.7405627, 0.7267445, 0.7054886, 0.687097, 0.6578311, 0.6082509, 0.5298675, 0.475823, 0.4094206, 0.3290132 };
   Double_t divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_felx3001[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_fely3001[30] = { 0.0006721855, 0.001013374, 0.001215027, 0.001410765, 0.001579363, 0.001757977, 0.001924719, 0.002104064, 0.002280693, 0.00243288, 0.002563725, 0.002773462, 0.002951632, 0.00312319, 0.003306858, 0.002502427, 0.002732886,
   0.002922581, 0.003155213, 0.003375395, 0.003280251, 0.003548429, 0.003875844, 0.00414008, 0.003265897, 0.003683607, 0.004150351, 0.00335666, 0.003248574, 0.003086877 };
   Double_t divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_fehx3001[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_fehy3001[30] = { 0.0006683837, 0.001005599, 0.001204423, 0.001397465, 0.001563589, 0.001739813, 0.001904277, 0.002081261, 0.002255967, 0.002405387, 0.002534008, 0.002741281, 0.002918329, 0.00308684, 0.003268629, 0.0024821, 0.002710212,
   0.0028971, 0.003126864, 0.003344489, 0.003253746, 0.003520154, 0.003846622, 0.004110737, 0.003251113, 0.003671412, 0.004146258, 0.003358832, 0.003256447, 0.003101606 };
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(30,divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_fx3001,divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_fy3001,divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_felx3001,divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_fehx3001,divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_fely3001,divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF_fehy3001);
   grae->SetName("divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF");
   grae->SetTitle("");

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#7a21dd");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#7a21dd");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(20);
   //grae->SetMarkerSize(2.5);
   grae->SetMarkerSize(3);
   
   TH1F *Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001 = new TH1F("Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001","",100,0,120);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->SetMinimum(0);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->SetMaximum(1.5);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->SetDirectory(nullptr);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->SetLineColor(ci);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetXaxis()->SetTitle("Gen.-level muon track d_{0} [cm]");
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetXaxis()->SetNdivisions(509);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetXaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetXaxis()->SetTitleSize(0.045);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetXaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetXaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetYaxis()->SetTitle("L1T efficiency");
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetYaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetYaxis()->SetTitleSize(0.045);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetYaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetYaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetZaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetZaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF3001);
   
   grae->Draw("ap");
   TGaxis *gaxis = new TGaxis(0,1.5,120,1.5,0,120,510,"-U");
   gaxis->SetLabelOffset(0.005);
   gaxis->SetLabelSize(0.04);
   gaxis->SetTickSize(0.03);
   gaxis->SetGridLength(0);
   gaxis->SetTitleOffset(1);
   gaxis->SetTitleSize(0.04);
   gaxis->SetTitleColor(1);
   gaxis->SetTitleFont(62);
   gaxis->Draw();
   gaxis = new TGaxis(120,0,120,1.5,0,1.5,510,"+U");
   gaxis->SetLabelOffset(0.005);
   gaxis->SetLabelSize(0.04);
   gaxis->SetTickSize(0.03);
   gaxis->SetGridLength(0);
   gaxis->SetTitleOffset(1);
   gaxis->SetTitleSize(0.04);
   gaxis->SetTitleColor(1);
   gaxis->SetTitleFont(62);
   gaxis->Draw();
   
   Double_t divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_fx3002[30] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 36,
   40, 44, 48, 52.5, 57.5, 62.5, 67.5, 75, 85, 95, 110, 135, 175 };
   Double_t divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_fy3002[30] = { 0.9060805, 0.8955763, 0.8908983, 0.8835245, 0.8787879, 0.871093, 0.8642703, 0.857504, 0.8495857, 0.8482719, 0.8488356, 0.8401946, 0.8308947, 0.8282786, 0.8189744, 0.8101025, 0.7970186,
   0.7758742, 0.7602549, 0.7312815, 0.6834198, 0.6191359, 0.5167336, 0.3860794, 0.226888, 0.1247694, 0.07155963, 0.04786332, 0.02961748, 0.01937636 };
   Double_t divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_felx3002[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_fely3002[30] = { 0.0006423958, 0.0009766618, 0.001171607, 0.001364211, 0.001524333, 0.001698877, 0.001861588, 0.002031216, 0.00219566, 0.002334837, 0.002437656, 0.002636985, 0.002799352, 0.002960664, 0.003147358, 0.002377513, 0.002621301,
   0.002893413, 0.003164945, 0.003477505, 0.003475288, 0.003855915, 0.004230858, 0.004320709, 0.002866459, 0.00247744, 0.00212932, 0.001427663, 0.001114913, 0.0009019949 };
   Double_t divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_fehx3002[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_fehy3002[30] = { 0.0006384959, 0.0009686992, 0.001160733, 0.001350567, 0.001508098, 0.00168015, 0.001840474, 0.002007548, 0.002169812, 0.002305957, 0.00240606, 0.00260254, 0.002763271, 0.002921145, 0.003105729, 0.002355109, 0.002596507,
   0.002867303, 0.003136833, 0.003449512, 0.003455089, 0.003841069, 0.00422848, 0.004338567, 0.002892067, 0.002519864, 0.002188481, 0.00146863, 0.001156316, 0.0009440862 };
   grae = new TGraphAsymmErrors(30,divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_fx3002,divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_fy3002,divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_felx3002,divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_fehx3002,divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_fely3002,divide_h_dxy_trg_BMTF_by_h_dxy_BMTF_fehy3002);
   grae->SetName("divide_h_dxy_trg_BMTF_by_h_dxy_BMTF");
   grae->SetTitle("");

   ci = TColor::GetColor("#7a21dd");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#7a21dd");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(24);
   //grae->SetMarkerSize(2.5);
   grae->SetMarkerSize(3);
   
   TH1F *Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002 = new TH1F("Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002","",100,0,220);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->SetMinimum(0);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->SetMaximum(1.5);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->SetDirectory(nullptr);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->SetLineColor(ci);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetXaxis()->SetTitle("d_{0} (gen #mu) [cm]");
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetXaxis()->SetRange(1,55);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetXaxis()->SetNdivisions(509);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetXaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetXaxis()->SetTitleSize(1);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetXaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetXaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetYaxis()->SetTitle("L1T efficiency");
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetYaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetYaxis()->SetTitleSize(1);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetYaxis()->SetTitleOffset(1.35);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetYaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetZaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetZaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_divide_h_dxy_trg_BMTF_by_h_dxy_BMTF3002);
   
   grae->Draw("s p");
   
   TLegend *leg = new TLegend(0.37,0.55,0.88,0.75,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.045);
   leg->SetLineColor(1);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("divide_h_dxy_trg_BMTF_by_h_dxy_BMTF","Beam axis-constrained","pe");

   ci = TColor::GetColor("#7a21dd");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#7a21dd");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(24);
   //entry->SetMarkerSize(2.5);
   entry->SetMarkerSize(3);
   entry->SetTextFont(42);
   entry=leg->AddEntry("divide_h_dxy_NN_trg_BMTF_by_h_dxy_BMTF","Beam axis-unconstrained","pe");

   ci = TColor::GetColor("#7a21dd");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#7a21dd");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(20);
   //entry->SetMarkerSize(2.5);
   entry->SetMarkerSize(3);
   entry->SetTextFont(42);
   leg->Draw();
   TLatex *   tex = new TLatex(0.42,0.77,"#font[42]{|#eta^{gen}_{st2}| < 0.83}");
   tex->SetNDC();
   tex->SetTextSize(0.04);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.42,0.83,"#font[42]{L1T p_{T} > 10 GeV, gen p_{T} > 15 GeV}");
   tex->SetNDC();
   tex->SetTextSize(0.04);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.1,0.92,"#bf{ #font[61]{CMS}} #font[52]{Simulation}");
   tex->SetNDC();
   tex->SetTextSize(0.045);
   tex->SetLineWidth(2);
   tex->Draw();
      tex = new TLatex(0.75,0.92,"#font[42]{(13.6 TeV)}");
   tex->SetNDC();
   tex->SetTextSize(0.04);
   tex->SetLineWidth(2);
   tex->Draw();
   c->Modified();
   c->SetSelected(c);
   c->SaveAs("Figure_039-a.pdf");
}
