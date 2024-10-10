USE [CornSilo]
GO

/****** Object:  Table [dbo].[LicensePlateDetection]    Script Date: 02/01/2020 19:42:33 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[LicensePlateDetection](
	[RecordedTime] [datetime] NOT NULL,
	[Location] [nvarchar](10) NOT NULL,
	[LicensePlateNumber] [nvarchar](10) NULL,
	[NO1Confident] [float] NULL,
	[NO2Confident] [float] NULL,
	[NO3Confident] [float] NULL,
	[NO4Confident] [float] NULL,
	[NO5Confident] [float] NULL,
	[NO6Confident] [float] NULL,
	[LicensePlateText] [nvarchar](5) NULL,
	[TextConfident] [float] NULL,
	[TruckType] [nvarchar](10) NULL,
	[Direction] [tinyint] NULL,
 CONSTRAINT [PK_LicensePlateDetection] PRIMARY KEY CLUSTERED 
(
	[RecordedTime] ASC,
	[Location] ASC
)WITH (PAD_INDEX  = OFF, STATISTICS_NORECOMPUTE  = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS  = ON, ALLOW_PAGE_LOCKS  = ON) ON [PRIMARY]
) ON [PRIMARY]

GO


