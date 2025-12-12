#ifdef __CLING__
#pragma cling optimize(0)
#endif
void Figure39_upper_right_canvas()
{
//=========Macro generated from canvas: c/
//=========  (Thu Nov 20 14:00:47 2025) by ROOT version 6.30/07
   TCanvas *c = new TCanvas("c", "",1,1,1280,1000);
   c->SetHighLightColor(2);
   c->Range(-15,-0.1875,135,1.6875);
   c->SetFillColor(0);
   c->SetBorderMode(0);
   c->SetBorderSize(2);
   c->SetFrameBorderMode(0);
   c->SetFrameBorderMode(0);
   
   Double_t divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_fx3003[30] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 36,
   40, 44, 48, 52.5, 57.5, 62.5, 67.5, 75, 85, 95, 110, 135, 175 };
   Double_t divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_fy3003[30] = { 0.710563, 0.7027479, 0.695033, 0.6869798, 0.6967837, 0.6869565, 0.6800692, 0.6811361, 0.662663, 0.6545667, 0.6525104, 0.6400526, 0.640241, 0.6116848, 0.6216381, 0.603542, 0.5907667,
   0.5957608, 0.5691569, 0.5539318, 0.5289757, 0.4996025, 0.4786571, 0.44658, 0.4021818, 0.3896226, 0.3085862, 0.2556739, 0.1572917, 0.0692354 };
   Double_t divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_felx3003[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_fely3003[30] = { 0.001638428, 0.002518862, 0.003092625, 0.003577786, 0.004003884, 0.00442917, 0.004922865, 0.005274494, 0.005819176, 0.006259437, 0.006664952, 0.007248842, 0.007603894, 0.008199642, 0.0086678, 0.006582905, 0.007229538,
   0.007945213, 0.008747252, 0.009463681, 0.009339171, 0.01016466, 0.01116671, 0.01206814, 0.009494226, 0.01077152, 0.01192861, 0.009502385, 0.008384208, 0.006259486 };
   Double_t divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_fehx3003[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_fehy3003[30] = { 0.001632971, 0.002506678, 0.003075248, 0.003555842, 0.003974514, 0.004395657, 0.004883508, 0.00522903, 0.005770918, 0.006207027, 0.006606555, 0.007186462, 0.007535225, 0.008137927, 0.008592123, 0.006546106, 0.007191063,
   0.00789611, 0.008705036, 0.009425454, 0.009319291, 0.01016498, 0.01118762, 0.01212996, 0.009566734, 0.01087793, 0.01218245, 0.009734236, 0.008752274, 0.006803068 };
   TGraphAsymmErrors *grae = new TGraphAsymmErrors(30,divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_fx3003,divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_fy3003,divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_felx3003,divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_fehx3003,divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_fely3003,divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF_fehy3003);
   grae->SetName("divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF");
   grae->SetTitle("");

   Int_t ci;      // for color index setting
   TColor *color; // for color definition with alpha
   ci = TColor::GetColor("#f89c20");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#f89c20");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(20);
   //grae->SetMarkerSize(2.5);
   grae->SetMarkerSize(3);
   
   TH1F *Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003 = new TH1F("Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003","",100,0,120);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->SetMinimum(0);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->SetMaximum(1.5);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->SetDirectory(nullptr);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->SetLineColor(ci);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetXaxis()->SetTitle("Gen.-level muon track d_{0} [cm]");
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetXaxis()->SetNdivisions(509);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetXaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetXaxis()->SetTitleSize(0.045);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetXaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetXaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetYaxis()->SetTitle("L1T efficiency");
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetYaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetYaxis()->SetTitleSize(0.045);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetYaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetYaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetZaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetZaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF3003);
   
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
   
   Double_t divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_fx3004[30] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 36,
   40, 44, 48, 52.5, 57.5, 62.5, 67.5, 75, 85, 95, 110, 135, 175 };
   Double_t divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_fy3004[30] = { 0.9260306, 0.9201912, 0.9135742, 0.9087364, 0.9040296, 0.8970719, 0.8862949, 0.8836661, 0.8711, 0.860823, 0.8618347, 0.8460189, 0.8257831, 0.8005435, 0.8059291, 0.7613537, 0.7113014,
   0.6701993, 0.6216622, 0.5845511, 0.5272911, 0.4550874, 0.3784173, 0.3001696, 0.1767273, 0.075, 0.02065849, 0.008800371, 0.008854167, 0.007224564 };
   Double_t divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_felx3004[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_fely3004[30] = { 0.0009522577, 0.001508681, 0.001909398, 0.002250295, 0.002600042, 0.00294248, 0.003396828, 0.003681063, 0.004184018, 0.004621751, 0.004904729, 0.005532375, 0.006088003, 0.006805691, 0.00716156, 0.005778236, 0.006698322,
   0.007637137, 0.0085863, 0.009394667, 0.009340335, 0.01010634, 0.01079904, 0.01106116, 0.007334327, 0.005748531, 0.003607151, 0.001995784, 0.002120721, 0.002052383 };
   Double_t divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_fehx3004[30] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
   2, 2, 2, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 10, 15, 25 };
   Double_t divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_fehy3004[30] = { 0.0009411849, 0.001483327, 0.001872381, 0.002202086, 0.002539426, 0.002870925, 0.003311961, 0.00358425, 0.004073348, 0.004498791, 0.004765432, 0.005377484, 0.005927808, 0.00663901, 0.006970463, 0.005685167, 0.006608644,
   0.007549805, 0.008512005, 0.009334727, 0.009321611, 0.01014275, 0.01091827, 0.01129287, 0.007575261, 0.006165904, 0.004281184, 0.002503198, 0.002694583, 0.002731532 };
   grae = new TGraphAsymmErrors(30,divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_fx3004,divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_fy3004,divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_felx3004,divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_fehx3004,divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_fely3004,divide_h_dxy_trg_OMTF_by_h_dxy_OMTF_fehy3004);
   grae->SetName("divide_h_dxy_trg_OMTF_by_h_dxy_OMTF");
   grae->SetTitle("");

   ci = TColor::GetColor("#f89c20");
   grae->SetLineColor(ci);

   ci = TColor::GetColor("#f89c20");
   grae->SetMarkerColor(ci);
   grae->SetMarkerStyle(24);
   //grae->SetMarkerSize(2.5);
   grae->SetMarkerSize(3);
   
   TH1F *Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004 = new TH1F("Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004","",100,0,220);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->SetMinimum(0);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->SetMaximum(1.5);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->SetDirectory(nullptr);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->SetStats(0);

   ci = TColor::GetColor("#000099");
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->SetLineColor(ci);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetXaxis()->SetTitle("d_{0} (gen #mu) [cm]");
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetXaxis()->SetRange(1,55);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetXaxis()->SetNdivisions(509);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetXaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetXaxis()->SetTitleSize(1);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetXaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetXaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetYaxis()->SetTitle("L1T efficiency");
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetYaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetYaxis()->SetTitleSize(1);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetYaxis()->SetTitleOffset(1.35);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetYaxis()->SetTitleFont(42);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetZaxis()->SetLabelFont(42);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetZaxis()->SetTitleOffset(1);
   Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004->GetZaxis()->SetTitleFont(42);
   grae->SetHistogram(Graph_divide_h_dxy_trg_OMTF_by_h_dxy_OMTF3004);
   
   grae->Draw("s p");
   
   TLegend *leg = new TLegend(0.37,0.55,0.88,0.75,NULL,"brNDC");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.045);
   leg->SetLineColor(1);
   leg->SetLineStyle(1);
   leg->SetLineWidth(1);
   leg->SetFillColor(0);
   leg->SetFillStyle(0);
   TLegendEntry *entry=leg->AddEntry("divide_h_dxy_trg_OMTF_by_h_dxy_OMTF","Beam axis-constrained","pe");

   ci = TColor::GetColor("#f89c20");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#f89c20");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(24);
   //entry->SetMarkerSize(2.5);
   entry->SetMarkerSize(3);
   entry->SetTextFont(42);
   entry=leg->AddEntry("divide_h_dxy_NN_trg_OMTF_by_h_dxy_OMTF","Beam axis-unconstrained","pe");

   ci = TColor::GetColor("#f89c20");
   entry->SetLineColor(ci);
   entry->SetLineStyle(1);
   entry->SetLineWidth(1);

   ci = TColor::GetColor("#f89c20");
   entry->SetMarkerColor(ci);
   entry->SetMarkerStyle(20);
   //entry->SetMarkerSize(2.5);
   entry->SetMarkerSize(3);
   entry->SetTextFont(42);
   leg->Draw();
   TLatex *   tex = new TLatex(0.42,0.77,"#font[42]{0.83 < |#eta^{gen}_{st2}| < 1.24}");
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
   c->SaveAs("Figure_039-b.pdf");
}
